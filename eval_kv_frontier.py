from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_CACHE", str(Path(".hf-cache/hub").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(".hf-cache/transformers").resolve()))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Item:
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    query_position_index: int
    target_position_index: int
    target_record: str
    oracle_records: tuple[str, ...]
    lexical_query: str
    record_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality under full/truncated/streaming prompt policies."
    )
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        choices=[
            "kv",
            "alias_bridge",
            "paraphrase_bridge",
            "varied_bridge",
            "format_bridge",
            "id_bridge",
            "descriptor_bridge",
            "shuffled_descriptor_bridge",
            "heldout_descriptor_bridge",
            "heldout_descriptor_bridge_v2",
            "heldout_descriptor_bridge_hard_negative",
        ],
        default="kv",
        help="Synthetic task family to evaluate.",
    )
    parser.add_argument(
        "--target-position",
        choices=["prefix", "middle", "recent", "random"],
        default="prefix",
        help="Where to place the answer-bearing record within the record list.",
    )
    parser.add_argument("--truncate-tokens", type=int, default=192)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--recent-tokens", type=int, default=160)
    parser.add_argument(
        "--lexical-records",
        type=int,
        default=1,
        help="Number of record lines to retain for lexical retrieval.",
    )
    parser.add_argument(
        "--prefilter-candidates",
        type=int,
        default=16,
        help="Number of descriptor-prefiltered candidate memos to rerank in prefilter_lm_bridge.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--scoring", choices=["generate", "rank"], default="generate")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--diagnose-final-scores",
        action="store_true",
        help="Emit descriptor-prefilter final-selection score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-contrastive-scores",
        action="store_true",
        help="Emit contrastive descriptor verifier score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-fewshot-scores",
        action="store_true",
        help="Emit few-shot descriptor translation score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-forced-choice-scores",
        action="store_true",
        help="Emit forced-choice descriptor label score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-pairwise-tournament-scores",
        action="store_true",
        help="Emit pairwise descriptor tournament traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-induced-map-scores",
        action="store_true",
        help="Emit in-context induced descriptor-map score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-external-support-induced-map-scores",
        action="store_true",
        help="Emit task-external support induced descriptor-map score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=5,
        help="Number of top candidates to print per diagnostic scorer.",
    )
    parser.add_argument("--metrics-path", default=os.environ.get("AAD_METRICS_PATH"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def target_insert_index(
    target_position: str,
    n_records: int,
    rng: random.Random,
) -> int:
    if target_position == "prefix":
        return 0
    if target_position == "middle":
        return n_records // 2
    if target_position == "recent":
        return n_records - 1
    if target_position == "random":
        return rng.randrange(n_records)
    raise ValueError(f"unknown target position: {target_position}")


def build_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_key = f"K{idx:03d}"
        target_value = f"{rng.randrange(100, 999)}"
        used_values = {target_value}
        candidate_values = [target_value]
        target_record = f"Record 000: {target_key} has value {target_value}."
        distractor_records = []
        for distractor_idx in range(n_distractors):
            key = f"D{idx:03d}_{distractor_idx:03d}"
            value = f"{rng.randrange(100, 999)}"
            while value in used_values:
                value = f"{rng.randrange(100, 999)}"
            used_values.add(value)
            candidate_values.append(value)
            distractor_records.append(
                f"Record {distractor_idx + 1:03d}: {key} has value {value}."
            )
        rng.shuffle(distractor_records)
        records = list(distractor_records)
        insert_idx = target_insert_index(
            target_position,
            n_records=n_distractors + 1,
            rng=rng,
        )
        records.insert(insert_idx, target_record)
        prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(records)
            + f"\nQuestion: What value belongs to {target_key}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_value,
                candidates=tuple(candidate_values),
                query_position_index=insert_idx,
                target_position_index=insert_idx,
                target_record=target_record,
                oracle_records=(target_record,),
                lexical_query=target_key,
                record_count=len(records),
            )
        )
    return items


ALIAS_NAMES = [
    ("Harbor Cartographer", "Mira Solen"),
    ("Glass Archivist", "Taro Venn"),
    ("Copper Botanist", "Leah Orin"),
    ("Signal Mason", "Niko Vale"),
    ("Ivory Surveyor", "Rhea Kato"),
    ("Amber Mechanic", "Sana Ives"),
    ("Blue Notary", "Oren Lark"),
    ("Cedar Analyst", "Vera Noll"),
    ("Winter Broker", "Jules Mori"),
    ("Quiet Pilot", "Kira Sato"),
    ("Silver Curator", "Milo Ren"),
    ("North Interpreter", "Ada Pell"),
]


def alias_name_pair(index: int) -> tuple[str, str]:
    if index < len(ALIAS_NAMES):
        return ALIAS_NAMES[index]
    return (f"Relay Specialist {index:03d}", f"Person {index:03d}")


def build_alias_bridge_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_alias, target_name = alias_name_pair(idx)
        target_code = f"{rng.randrange(100, 999)}"
        used_codes = {target_code}
        candidate_values = [target_code]
        alias_record = f"Alias 000: {target_alias} refers to {target_name}."
        profile_record = f"Profile 000: {target_name} uses access code {target_code}."

        distractor_records = []
        for distractor_idx in range(n_distractors):
            alias, name = alias_name_pair(idx + n_items + distractor_idx)
            code = f"{rng.randrange(100, 999)}"
            while code in used_codes:
                code = f"{rng.randrange(100, 999)}"
            used_codes.add(code)
            candidate_values.append(code)
            distractor_records.append(
                f"Alias {distractor_idx + 1:03d}: {alias} refers to {name}."
            )
            distractor_records.append(
                f"Profile {distractor_idx + 1:03d}: {name} uses access code {code}."
            )

        rng.shuffle(distractor_records)
        records = list(distractor_records)
        insert_idx = target_insert_index(
            target_position,
            n_records=len(distractor_records) + 2,
            rng=rng,
        )
        records[insert_idx:insert_idx] = [alias_record, profile_record]
        prompt = (
            "You are given alias and profile records. Answer with only the three digit access code.\n"
            + "\n".join(records)
            + f"\nQuestion: What access code belongs to {target_alias}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_code,
                candidates=tuple(candidate_values),
                query_position_index=insert_idx,
                target_position_index=insert_idx,
                target_record=profile_record,
                oracle_records=(alias_record, profile_record),
                lexical_query=target_alias,
                record_count=len(records),
            )
        )
    return items


def build_paraphrase_bridge_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_alias, target_name = alias_name_pair(idx)
        target_code = f"{rng.randrange(100, 999)}"
        used_codes = {target_code}
        candidate_values = [target_code]
        alias_record = (
            f"Memo 000A: In field notes, the title \"{target_alias}\" is used "
            f"for {target_name} during audits."
        )
        profile_record = (
            f"Memo 000B: During badge setup, {target_name} received vault code "
            f"{target_code}."
        )

        distractor_records = []
        for distractor_idx in range(n_distractors):
            alias, name = alias_name_pair(idx + n_items + distractor_idx)
            code = f"{rng.randrange(100, 999)}"
            while code in used_codes:
                code = f"{rng.randrange(100, 999)}"
            used_codes.add(code)
            candidate_values.append(code)
            record_id = distractor_idx + 1
            distractor_records.append(
                f"Memo {record_id:03d}A: In field notes, the title \"{alias}\" "
                f"is used for {name} during audits."
            )
            distractor_records.append(
                f"Memo {record_id:03d}B: During badge setup, {name} received "
                f"vault code {code}."
            )

        rng.shuffle(distractor_records)
        records = list(distractor_records)
        insert_idx = target_insert_index(
            target_position,
            n_records=len(distractor_records) + 2,
            rng=rng,
        )
        records[insert_idx:insert_idx] = [alias_record, profile_record]
        prompt = (
            "You are given personnel memos. Answer with only the three digit vault code.\n"
            + "\n".join(records)
            + f"\nQuestion: Which vault code is assigned to {target_alias}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_code,
                candidates=tuple(candidate_values),
                query_position_index=insert_idx,
                target_position_index=insert_idx,
                target_record=profile_record,
                oracle_records=(alias_record, profile_record),
                lexical_query=target_alias,
                record_count=len(records),
            )
        )
    return items


ALIAS_TEMPLATES = [
    'Memo {record_id:03d}A: Audit staff use the title "{alias}" when writing about {name}.',
    'Memo {record_id:03d}A: The field title "{alias}" points auditors toward {name}.',
    'Memo {record_id:03d}A: In dispatch notes, {name} is the person behind "{alias}".',
    'Memo {record_id:03d}A: Badge reviewers file "{alias}" as a title for {name}.',
]

PROFILE_TEMPLATES = [
    "Memo {record_id:03d}B: Badge setup gave {name} vault code {code}.",
    "Memo {record_id:03d}B: The vault code recorded for {name} is {code}.",
    "Memo {record_id:03d}B: Security assigned {code} as the vault code for {name}.",
    "Memo {record_id:03d}B: During onboarding, {name} was issued vault code {code}.",
]


def varied_alias_record(record_id: int, alias: str, name: str, rng: random.Random) -> str:
    template = rng.choice(ALIAS_TEMPLATES)
    return template.format(record_id=record_id, alias=alias, name=name)


def varied_profile_record(
    record_id: int,
    name: str,
    code: str,
    rng: random.Random,
) -> str:
    template = rng.choice(PROFILE_TEMPLATES)
    return template.format(record_id=record_id, name=name, code=code)


def build_varied_bridge_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
    lowercase_names: bool = False,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_alias, target_name = alias_name_pair(idx)
        if lowercase_names:
            target_name = target_name.lower()
        target_code = f"{rng.randrange(100, 999)}"
        used_codes = {target_code}
        candidate_values = [target_code]
        alias_record = varied_alias_record(0, target_alias, target_name, rng)
        profile_record = varied_profile_record(0, target_name, target_code, rng)

        distractor_records = []
        for distractor_idx in range(n_distractors):
            alias, name = alias_name_pair(idx + n_items + distractor_idx)
            if lowercase_names:
                name = name.lower()
            code = f"{rng.randrange(100, 999)}"
            while code in used_codes:
                code = f"{rng.randrange(100, 999)}"
            used_codes.add(code)
            candidate_values.append(code)
            record_id = distractor_idx + 1
            distractor_records.append(varied_alias_record(record_id, alias, name, rng))
            distractor_records.append(varied_profile_record(record_id, name, code, rng))

        rng.shuffle(distractor_records)
        records = list(distractor_records)
        insert_idx = target_insert_index(
            target_position,
            n_records=len(distractor_records) + 2,
            rng=rng,
        )
        records[insert_idx:insert_idx] = [alias_record, profile_record]
        prompt = (
            "You are given personnel memos. Answer with only the three digit vault code.\n"
            + "\n".join(records)
            + f"\nQuestion: Which vault code is assigned to {target_alias}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_code,
                candidates=tuple(candidate_values),
                query_position_index=insert_idx,
                target_position_index=insert_idx,
                target_record=profile_record,
                oracle_records=(alias_record, profile_record),
                lexical_query=target_alias,
                record_count=len(records),
            )
        )
    return items


def employee_id(index: int) -> str:
    letter_a = chr(ord("A") + (index % 26))
    letter_b = chr(ord("A") + ((index // 26) % 26))
    return f"{letter_a}{letter_b}-{100 + index:03d}"


def build_id_bridge_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_alias, _target_name = alias_name_pair(idx)
        target_employee_id = employee_id(idx)
        target_staff_number = normalize_identifier(target_employee_id)
        target_code = f"{rng.randrange(100, 999)}"
        used_codes = {target_code}
        candidate_values = [target_code]
        alias_record = (
            f'Memo 000A: The field title "{target_alias}" maps to employee id '
            f"{target_employee_id}."
        )
        profile_record = (
            f"Memo 000B: Credential file for staff number {target_staff_number} "
            f"lists vault code {target_code}."
        )

        distractor_records = []
        for distractor_idx in range(n_distractors):
            alias, _name = alias_name_pair(idx + n_items + distractor_idx)
            current_id = employee_id(idx + n_items + distractor_idx)
            staff_number = normalize_identifier(current_id)
            code = f"{rng.randrange(100, 999)}"
            while code in used_codes:
                code = f"{rng.randrange(100, 999)}"
            used_codes.add(code)
            candidate_values.append(code)
            record_id = distractor_idx + 1
            distractor_records.append(
                f'Memo {record_id:03d}A: The field title "{alias}" maps to '
                f"employee id {current_id}."
            )
            distractor_records.append(
                f"Memo {record_id:03d}B: Credential file for staff number "
                f"{staff_number} lists vault code {code}."
            )

        rng.shuffle(distractor_records)
        records = list(distractor_records)
        insert_idx = target_insert_index(
            target_position,
            n_records=len(distractor_records) + 2,
            rng=rng,
        )
        records[insert_idx:insert_idx] = [alias_record, profile_record]
        prompt = (
            "You are given personnel memos. Answer with only the three digit vault code.\n"
            + "\n".join(records)
            + f"\nQuestion: Which vault code is assigned to {target_alias}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_code,
                candidates=tuple(candidate_values),
                query_position_index=insert_idx,
                target_position_index=insert_idx,
                target_record=profile_record,
                oracle_records=(alias_record, profile_record),
                lexical_query=target_alias,
                record_count=len(records),
            )
        )
    return items


DESCRIPTOR_PAIRS = [
    ("coastal route analyst", "nautical chart planner"),
    ("glass archive reviewer", "crystal record curator"),
    ("copper garden inspector", "bronze plant examiner"),
    ("radio tower builder", "signal mast constructor"),
    ("ivory terrain scout", "pale landscape surveyor"),
    ("amber engine repairer", "golden motor mechanic"),
    ("blue contract witness", "azure document notary"),
    ("cedar data reviewer", "timber report analyst"),
    ("winter market broker", "frost exchange agent"),
    ("quiet flight operator", "silent aircraft pilot"),
    ("silver museum keeper", "metal gallery curator"),
    ("northern speech translator", "polar language interpreter"),
]

DESCRIPTOR_FAMILIES = [
    (("coastal", "route", "analyst"), ("nautical", "chart", "planner")),
    (("glass", "archive", "reviewer"), ("crystal", "record", "curator")),
    (("copper", "garden", "inspector"), ("bronze", "plant", "examiner")),
    (("radio", "tower", "builder"), ("signal", "mast", "constructor")),
    (("ivory", "terrain", "scout"), ("pale", "landscape", "surveyor")),
    (("amber", "engine", "repairer"), ("golden", "motor", "mechanic")),
    (("blue", "contract", "witness"), ("azure", "document", "notary")),
    (("cedar", "data", "reviewer"), ("timber", "report", "analyst")),
    (("winter", "market", "broker"), ("frost", "exchange", "agent")),
    (("quiet", "flight", "operator"), ("silent", "aircraft", "pilot")),
    (("silver", "museum", "keeper"), ("metal", "gallery", "curator")),
    (("northern", "speech", "translator"), ("polar", "language", "interpreter")),
    (("river", "cargo", "scheduler"), ("stream", "freight", "planner")),
    (("marble", "harbor", "auditor"), ("stone", "port", "reviewer")),
    (("violet", "signal", "mapper"), ("purple", "beacon", "cartographer")),
    (("forest", "sample", "collector"), ("woodland", "specimen", "gatherer")),
    (("solar", "panel", "tester"), ("sunlit", "array", "examiner")),
    (("linen", "ledger", "keeper"), ("fabric", "registry", "curator")),
    (("urban", "bridge", "designer"), ("city", "span", "architect")),
    (("opal", "clinic", "observer"), ("gem", "infirmary", "watcher")),
    (("magnetic", "rail", "planner"), ("lodestone", "track", "scheduler")),
    (("pearl", "tunnel", "scanner"), ("shell", "passage", "inspector")),
    (("prairie", "weather", "reader"), ("grassland", "climate", "interpreter")),
    (("coral", "reef", "warden"), ("rose", "shoal", "guardian")),
    (("granite", "quarry", "measurer"), ("rock", "mine", "surveyor")),
    (("velvet", "stage", "manager"), ("soft", "theater", "coordinator")),
    (("lunar", "shadow", "tracker"), ("moonlit", "shade", "monitor")),
    (("orchard", "harvest", "counter"), ("grove", "yield", "tallier")),
    (("platinum", "circuit", "tester"), ("bright", "wire", "examiner")),
    (("desert", "well", "finder"), ("arid", "spring", "locator")),
    (("ceramic", "kiln", "operator"), ("clay", "furnace", "handler")),
    (("meadow", "insect", "cataloger"), ("field", "beetle", "indexer")),
    (("basalt", "road", "inspector"), ("darkstone", "highway", "examiner")),
    (("rainfall", "canal", "planner"), ("stormwater", "channel", "designer")),
    (("saffron", "menu", "reviewer"), ("yellow", "recipe", "critic")),
    (("alpine", "rescue", "coordinator"), ("mountain", "recovery", "dispatcher")),
    (("paper", "treaty", "registrar"), ("parchment", "accord", "clerk")),
    (("electric", "gate", "maintainer"), ("charged", "portal", "repairer")),
    (("sandstone", "tower", "watcher"), ("tanrock", "spire", "observer")),
    (("willow", "water", "tester"), ("branch", "liquid", "examiner")),
    (("cerulean", "map", "librarian"), ("skyblue", "atlas", "archivist")),
    (("iron", "warehouse", "marshal"), ("metallic", "depot", "supervisor")),
    (("morning", "train", "dispatcher"), ("dawn", "locomotive", "router")),
    (("obsidian", "mirror", "polisher"), ("blackglass", "reflector", "finisher")),
]

HELDOUT_DESCRIPTOR_PAIRS = [
    ("storm damage assessor", "weather loss inspector"),
    ("hospital supply coordinator", "clinic inventory organizer"),
    ("factory safety examiner", "workshop hazard reviewer"),
    ("school meal planner", "cafeteria nutrition scheduler"),
    ("library event host", "bookhall program presenter"),
    ("airport baggage router", "terminal luggage dispatcher"),
    ("theater lighting arranger", "stage illumination designer"),
    ("farm soil tester", "cropland earth analyst"),
    ("hotel room allocator", "lodging suite assigner"),
    ("museum ticket counter", "gallery admission tallyer"),
    ("city permit clerk", "municipal license registrar"),
    ("harbor fuel monitor", "dock energy watcher"),
    ("bakery recipe tester", "pastry formula examiner"),
    ("court schedule keeper", "tribunal calendar custodian"),
    ("rail cargo inspector", "train freight auditor"),
    ("clinic appointment caller", "infirmary visit notifier"),
    ("warehouse shelf labeler", "depot rack marker"),
    ("garden irrigation setter", "nursery watering adjuster"),
    ("market price tracker", "bazaar cost monitor"),
    ("studio sound mixer", "recording audio balancer"),
    ("road sign painter", "highway marker colorist"),
    ("kitchen appliance checker", "cookroom device examiner"),
    ("camp route guide", "outdoor path navigator"),
    ("office mail sorter", "workplace parcel classifier"),
    ("aquarium tank cleaner", "fishroom basin maintainer"),
    ("pharmacy label verifier", "dispensary tag checker"),
    ("stadium seating mapper", "arena chair cartographer"),
    ("laundry fabric sorter", "washroom textile classifier"),
    ("postal route planner", "mailway delivery scheduler"),
    ("laboratory sample freezer", "research specimen chiller"),
    ("concert crowd counter", "music hall attendance tallyer"),
    ("bakery oven watcher", "pastry furnace monitor"),
    ("clinic wheelchair tracker", "infirmary mobility chair locator"),
    ("warehouse pallet mover", "depot skid transporter"),
    ("school textbook auditor", "classroom volume reviewer"),
    ("airport runway sweeper", "terminal airstrip cleaner"),
    ("market stall assigner", "bazaar booth allocator"),
    ("hotel linen counter", "lodging sheet tallyer"),
    ("garden seed cataloger", "nursery grain indexer"),
    ("studio camera tester", "recording lens examiner"),
    ("court evidence logger", "tribunal exhibit registrar"),
    ("harbor anchor inspector", "dock mooring examiner"),
    ("office printer maintainer", "workplace copier repairer"),
    ("rail signal watcher", "train beacon observer"),
]


DESCRIPTOR_ALIAS_TEMPLATES = [
    'Memo {record_id:03d}A: The field title "{alias}" is assigned to the {source}.',
    'Memo {record_id:03d}A: Dispatch index maps title "{alias}" onto the {source}.',
    'Memo {record_id:03d}A: Review notes route "{alias}" toward the {source}.',
    'Memo {record_id:03d}A: Catalog staff tag "{alias}" as the {source}.',
]

DESCRIPTOR_PROFILE_TEMPLATES = [
    "Memo {record_id:03d}B: Credential card for the {target} lists vault code {code}.",
    "Memo {record_id:03d}B: Vault roster: the {target} carries vault code {code}.",
    "Memo {record_id:03d}B: Access sheet names the {target} with vault code {code}.",
    "Memo {record_id:03d}B: Security note records vault code {code} for the {target}.",
]


def descriptor_alias_record(
    record_id: int,
    alias: str,
    source: str,
    rng: random.Random,
    varied_templates: bool,
) -> str:
    template = (
        rng.choice(DESCRIPTOR_ALIAS_TEMPLATES)
        if varied_templates
        else DESCRIPTOR_ALIAS_TEMPLATES[0]
    )
    return template.format(record_id=record_id, alias=alias, source=source)


def descriptor_profile_record(
    record_id: int,
    target: str,
    code: str,
    rng: random.Random,
    varied_templates: bool,
) -> str:
    template = (
        rng.choice(DESCRIPTOR_PROFILE_TEMPLATES)
        if varied_templates
        else DESCRIPTOR_PROFILE_TEMPLATES[0]
    )
    return template.format(record_id=record_id, target=target, code=code)


def descriptor_pair(index: int) -> tuple[str, str]:
    if index < len(DESCRIPTOR_FAMILIES):
        source, target = DESCRIPTOR_FAMILIES[index]
        return (" ".join(source), " ".join(target))
    family = DESCRIPTOR_FAMILIES[index % len(DESCRIPTOR_FAMILIES)]
    source, target = family
    prefix = [
        "backup",
        "reserve",
        "auxiliary",
        "alternate",
        "standby",
        "relief",
    ][index // len(DESCRIPTOR_FAMILIES) % 6]
    return (f"{prefix} {' '.join(source)}", f"{prefix} {' '.join(target)}")


def heldout_descriptor_pair(index: int) -> tuple[str, str]:
    if index < len(HELDOUT_DESCRIPTOR_PAIRS):
        return HELDOUT_DESCRIPTOR_PAIRS[index]
    source, target = HELDOUT_DESCRIPTOR_PAIRS[index % len(HELDOUT_DESCRIPTOR_PAIRS)]
    prefix = [
        "backup",
        "reserve",
        "auxiliary",
        "alternate",
        "standby",
        "relief",
    ][index // len(HELDOUT_DESCRIPTOR_PAIRS) % 6]
    return (f"{prefix} {source}", f"{prefix} {target}")


def hard_negative_descriptor_pairs(
    source_descriptor: str,
    target_descriptor: str,
) -> list[tuple[str, str]]:
    source_words = source_descriptor.split()
    target_words = target_descriptor.split()
    if len(source_words) < 3 or len(target_words) < 3:
        return []
    source_head, source_domain, source_role = source_words[:3]
    target_head, target_domain, target_role = target_words[:3]
    return [
        (
            f"{source_head} {source_domain} reviewer",
            f"{target_head} {target_domain} examiner",
        ),
        (
            f"{source_head} {target_domain} {source_role}",
            f"{target_head} {source_domain} {target_role}",
        ),
        (
            f"{target_head} {source_domain} {source_role}",
            f"{source_head} {target_domain} {target_role}",
        ),
        (
            f"regional {source_domain} {source_role}",
            f"regional {target_domain} {target_role}",
        ),
        (
            f"backup {source_head} {source_role}",
            f"backup {target_head} {target_role}",
        ),
        (
            f"{source_head} auxiliary {source_role}",
            f"{target_head} auxiliary {target_role}",
        ),
    ]


def build_descriptor_bridge_items(
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
    shuffle_profiles: bool = False,
    heldout_pairs: bool = False,
    varied_templates: bool = False,
    hard_negatives: bool = False,
) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for idx in range(n_items):
        target_alias, _target_name = alias_name_pair(idx)
        pair_fn = heldout_descriptor_pair if heldout_pairs else descriptor_pair
        source_descriptor, target_descriptor = pair_fn(idx)
        target_code = f"{rng.randrange(100, 999)}"
        used_codes = {target_code}
        candidate_values = [target_code]
        alias_record = descriptor_alias_record(
            0,
            target_alias,
            source_descriptor,
            rng,
            varied_templates,
        )
        profile_record = descriptor_profile_record(
            0,
            target_descriptor,
            target_code,
            rng,
            varied_templates,
        )

        distractor_pairs: list[tuple[str, str]] = []
        hard_pairs = (
            hard_negative_descriptor_pairs(source_descriptor, target_descriptor)
            if hard_negatives
            else []
        )
        for distractor_idx in range(n_distractors):
            alias, _name = alias_name_pair(idx + n_items + distractor_idx)
            if distractor_idx < len(hard_pairs):
                source, target = hard_pairs[distractor_idx]
            else:
                source, target = pair_fn(idx + n_items + distractor_idx)
            code = f"{rng.randrange(100, 999)}"
            while code in used_codes:
                code = f"{rng.randrange(100, 999)}"
            used_codes.add(code)
            candidate_values.append(code)
            record_id = distractor_idx + 1
            distractor_pairs.append(
                (
                    descriptor_alias_record(
                        record_id,
                        alias,
                        source,
                        rng,
                        varied_templates,
                    ),
                    descriptor_profile_record(
                        record_id,
                        target,
                        code,
                        rng,
                        varied_templates,
                    ),
                )
            )

        rng.shuffle(distractor_pairs)
        if shuffle_profiles:
            alias_records = [pair[0] for pair in distractor_pairs]
            profile_records = [pair[1] for pair in distractor_pairs]
            rng.shuffle(alias_records)
            rng.shuffle(profile_records)
            records = alias_records + profile_records
        else:
            records = [line for pair in distractor_pairs for line in pair]
        insert_idx = target_insert_index(
            target_position,
            n_records=len(records) + 2,
            rng=rng,
        )
        if shuffle_profiles:
            records.insert(insert_idx, alias_record)
            profile_insert_idx = target_insert_index(
                target_position,
                n_records=len(records) + 1,
                rng=rng,
            )
            if profile_insert_idx in {insert_idx - 1, insert_idx, insert_idx + 1}:
                profile_insert_idx = (insert_idx + max(3, len(records) // 3)) % (
                    len(records) + 1
                )
            records.insert(profile_insert_idx, profile_record)
            target_index = profile_insert_idx
            query_index = insert_idx
            oracle = (alias_record, profile_record)
        else:
            records[insert_idx:insert_idx] = [alias_record, profile_record]
            target_index = insert_idx
            query_index = insert_idx
            oracle = (alias_record, profile_record)
        prompt = (
            "You are given personnel memos. Answer with only the three digit vault code.\n"
            + "\n".join(records)
            + f"\nQuestion: Which vault code is assigned to {target_alias}?\nAnswer:"
        )
        items.append(
            Item(
                prompt=prompt,
                answer=target_code,
                candidates=tuple(candidate_values),
                query_position_index=query_index,
                target_position_index=target_index,
                target_record=profile_record,
                oracle_records=oracle,
                lexical_query=target_alias,
                record_count=len(records),
            )
        )
    return items


def build_task_items(
    task: str,
    n_items: int,
    n_distractors: int,
    seed: int,
    target_position: str,
) -> list[Item]:
    if task == "kv":
        return build_items(n_items, n_distractors, seed, target_position)
    if task == "alias_bridge":
        return build_alias_bridge_items(n_items, n_distractors, seed, target_position)
    if task == "paraphrase_bridge":
        return build_paraphrase_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
        )
    if task == "varied_bridge":
        return build_varied_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
        )
    if task == "format_bridge":
        return build_varied_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
            lowercase_names=True,
        )
    if task == "id_bridge":
        return build_id_bridge_items(n_items, n_distractors, seed, target_position)
    if task == "descriptor_bridge":
        return build_descriptor_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
        )
    if task == "shuffled_descriptor_bridge":
        return build_descriptor_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
            shuffle_profiles=True,
        )
    if task == "heldout_descriptor_bridge":
        return build_descriptor_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
            shuffle_profiles=True,
            heldout_pairs=True,
        )
    if task == "heldout_descriptor_bridge_v2":
        return build_descriptor_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
            shuffle_profiles=True,
            heldout_pairs=True,
            varied_templates=True,
        )
    if task == "heldout_descriptor_bridge_hard_negative":
        return build_descriptor_bridge_items(
            n_items,
            n_distractors,
            seed,
            target_position,
            shuffle_profiles=True,
            heldout_pairs=True,
            varied_templates=True,
            hard_negatives=True,
        )
    raise ValueError(f"unknown task: {task}")


def choose_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if name == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args: argparse.Namespace):
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    device = choose_device(args.device)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def policy_token_ids(
    tokenizer,
    item: Item,
    token_ids: list[int],
    policy: str,
    truncate_tokens: int,
    sink_tokens: int,
    recent_tokens: int,
    lexical_records: int,
) -> list[int]:
    if policy == "full":
        return token_ids
    if policy == "truncate":
        return token_ids[-truncate_tokens:]
    if policy == "streaming":
        if len(token_ids) <= sink_tokens + recent_tokens:
            return token_ids
        return token_ids[:sink_tokens] + token_ids[-recent_tokens:]
    if policy == "oracle_span":
        question = item.prompt.rsplit("\nQuestion:", maxsplit=1)[1]
        oracle_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(item.oracle_records)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(oracle_prompt, add_special_tokens=False)
    if policy == "lexical":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        if not selected:
            selected = record_lines[:1]
        selected = selected[: max(1, lexical_records)]
        lexical_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(lexical_prompt, add_special_tokens=False)
    if policy == "bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        linked_names = []
        for line in selected:
            match = re.search(r" refers to (.+)\.$", line)
            if match:
                linked_names.append(match.group(1))
        for name in linked_names:
            selected.extend(
                line
                for line in record_lines
                if line.startswith("Profile ") and name in line and line not in selected
            )
        if not selected:
            selected = record_lines[:1]
        bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(bridge_prompt, add_special_tokens=False)
    if policy == "prose_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        linked_names: list[str] = []
        for line in selected:
            fixed_match = re.search(r" refers to (.+)\.$", line)
            if fixed_match:
                linked_names.append(fixed_match.group(1))
                continue
            prose_match = re.search(
                r"\bis used for ([A-Z][A-Za-z]+ [A-Z][A-Za-z]+)\b",
                line,
            )
            if prose_match:
                linked_names.append(prose_match.group(1))
        for name in linked_names:
            selected.extend(
                line for line in record_lines if name in line and line not in selected
            )
        if not selected:
            selected = record_lines[:1]
        prose_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(prose_bridge_prompt, add_special_tokens=False)
    if policy == "entity_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        query_spans = set(re.findall(r"\b[A-Z][A-Za-z]+ [A-Z][A-Za-z]+\b", item.lexical_query))
        linked_names: list[str] = []
        for line in selected:
            for name in re.findall(r"\b[A-Z][A-Za-z]+ [A-Z][A-Za-z]+\b", line):
                if name not in query_spans and name not in linked_names:
                    linked_names.append(name)
        for name in linked_names:
            selected.extend(
                line for line in record_lines if name in line and line not in selected
            )
        if not selected:
            selected = record_lines[:1]
        entity_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(entity_bridge_prompt, add_special_tokens=False)
    if policy == "overlap_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        query_tokens = content_tokens(item.lexical_query)
        selected_tokens = set()
        for line in selected:
            selected_tokens.update(content_tokens(line))
        bridge_tokens = selected_tokens - query_tokens
        scored_lines: list[tuple[int, int, str]] = []
        for line_idx, line in enumerate(record_lines):
            if line in selected:
                continue
            overlap = len(bridge_tokens & content_tokens(line))
            if overlap:
                scored_lines.append((overlap, -line_idx, line))
        if scored_lines:
            scored_lines.sort(reverse=True)
            selected.append(scored_lines[0][2])
        if not selected:
            selected = record_lines[:1]
        overlap_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(overlap_bridge_prompt, add_special_tokens=False)
    if policy == "id_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        identifiers: list[str] = []
        for line in selected:
            for identifier in extract_identifiers(line):
                if identifier not in identifiers:
                    identifiers.append(identifier)
        for identifier in identifiers:
            selected.extend(
                line
                for line in record_lines
                if line not in selected and identifier in extract_identifiers(line)
            )
        if not selected:
            selected = record_lines[:1]
        id_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(id_bridge_prompt, add_special_tokens=False)
    if policy == "char_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        gram_weights = char_ngram_idf(record_lines)
        query_grams = bridge_char_grams(item.lexical_query)
        selected_grams: set[str] = set()
        for line in selected:
            selected_grams.update(bridge_char_grams(line))
        bridge_grams = selected_grams - query_grams
        scored_lines: list[tuple[float, int, str]] = []
        for line_idx, line in enumerate(record_lines):
            if line in selected:
                continue
            overlap = sum(
                gram_weights.get(gram, 0.0)
                for gram in bridge_grams & bridge_char_grams(line)
            )
            if overlap:
                scored_lines.append((overlap, -line_idx, line))
        if scored_lines:
            scored_lines.sort(reverse=True)
            selected.append(scored_lines[0][2])
        if not selected:
            selected = record_lines[:1]
        char_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(char_bridge_prompt, add_special_tokens=False)
    if policy == "neighbor_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected_indices = [
            line_idx
            for line_idx, line in enumerate(record_lines)
            if item.lexical_query in line
        ]
        selected: list[str] = []
        for line_idx in selected_indices:
            for neighbor_idx in (line_idx, line_idx + 1):
                if 0 <= neighbor_idx < len(record_lines):
                    line = record_lines[neighbor_idx]
                    if line not in selected:
                        selected.append(line)
        if not selected:
            selected = record_lines[:1]
        neighbor_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(neighbor_bridge_prompt, add_special_tokens=False)
    if policy == "semantic_bridge":
        records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
        record_lines = [
            line
            for line in records_block.splitlines()
            if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
        ]
        selected = [line for line in record_lines if item.lexical_query in line]
        query_tokens = semantic_tokens(item.lexical_query)
        selected_tokens: set[str] = set()
        target_descriptor_tokens: set[str] = set()
        descriptor_map = descriptor_phrase_map()
        for line in selected:
            selected_tokens.update(semantic_tokens(line))
            descriptor_match = re.search(r"\bassigned to the ([^.]+)\.", line)
            if descriptor_match:
                source_descriptor = descriptor_match.group(1).lower()
                target_descriptor = descriptor_map.get(source_descriptor)
                if target_descriptor:
                    target_descriptor_tokens.update(content_tokens(target_descriptor))
        bridge_tokens = target_descriptor_tokens or (selected_tokens - query_tokens)
        scored_lines: list[tuple[int, int, str]] = []
        candidate_lines = [
            line for line in record_lines if "vault code" in line.lower()
        ]
        if not candidate_lines:
            candidate_lines = record_lines
        for line_idx, line in enumerate(candidate_lines):
            if line in selected:
                continue
            line_tokens = (
                content_tokens(line) if target_descriptor_tokens else semantic_tokens(line)
            )
            overlap = len(bridge_tokens & line_tokens)
            if overlap:
                scored_lines.append((overlap, -line_idx, line))
        if scored_lines:
            scored_lines.sort(reverse=True)
            selected.append(scored_lines[0][2])
        if not selected:
            selected = record_lines[:1]
        semantic_bridge_prompt = (
            "You are given records. Answer with only the three digit value.\n"
            + "\n".join(selected)
            + "\nQuestion:"
            + question
        )
        return tokenizer.encode(semantic_bridge_prompt, add_special_tokens=False)
    raise ValueError(f"unknown policy: {policy}")


@torch.inference_mode()
def average_continuation_logprob(
    tokenizer,
    model,
    device: torch.device,
    prefix: str,
    continuation: str,
) -> float:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not prefix_ids or not continuation_ids:
        return float("-inf")
    input_ids = torch.tensor(
        [prefix_ids + continuation_ids],
        dtype=torch.long,
        device=device,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    log_probs = torch.log_softmax(outputs.logits[0], dim=-1)
    total = 0.0
    for offset, token_id in enumerate(continuation_ids):
        total += float(log_probs[len(prefix_ids) - 1 + offset, token_id].item())
    return total / len(continuation_ids)


def lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
) -> list[int]:
    records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
    record_lines = [
        line
        for line in records_block.splitlines()
        if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
    ]
    selected = [line for line in record_lines if item.lexical_query in line]
    if not selected:
        selected = record_lines[:1]
    candidate_lines = [
        line for line in record_lines if line not in selected and "vault code" in line.lower()
    ]
    if not candidate_lines:
        candidate_lines = [line for line in record_lines if line not in selected]
    prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {selected[0]}\n"
        "Credential memo:"
    )
    scored_lines: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + line,
        )
        scored_lines.append((score, -line_idx, line))
    if scored_lines:
        scored_lines.sort(reverse=True)
        selected.append(scored_lines[0][2])
    lm_bridge_prompt = (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join(selected)
        + "\nQuestion:"
        + question
    )
    return tokenizer.encode(lm_bridge_prompt, add_special_tokens=False)


def credential_descriptor(line: str) -> str:
    patterns = [
        r"\bCredential card for the (.+?) lists vault code\b",
        r"\bVault roster: the (.+?) carries vault code\b",
        r"\bAccess sheet names the (.+?) with vault code\b",
        r"\bSecurity note records vault code \d{3} for the (.+?)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return line


def assigned_descriptor(line: str) -> str:
    patterns = [
        r"\bassigned to the (.+?)\.",
        r"\bmaps title \"[^\"]+\" onto the (.+?)\.",
        r"\broute \"[^\"]+\" toward the (.+?)\.",
        r"\btag \"[^\"]+\" as the (.+?)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return line


def vault_code(line: str) -> str:
    match = re.search(r"\bvault code (\d{3})\b", line, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def memo_record_id(line: str) -> int | None:
    match = re.match(r"Memo (\d{3})[AB]:", line)
    if match:
        return int(match.group(1))
    return None


def query_and_credential_lines(item: Item) -> tuple[str, list[str], str, list[str]]:
    records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
    record_lines = [
        line
        for line in records_block.splitlines()
        if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
    ]
    selected = [line for line in record_lines if item.lexical_query in line]
    if not selected:
        selected = record_lines[:1]
    candidate_lines = [
        line for line in record_lines if line not in selected and "vault code" in line.lower()
    ]
    if not candidate_lines:
        candidate_lines = [line for line in record_lines if line not in selected]
    return question, record_lines, selected[0], candidate_lines


def phrase_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
) -> list[int]:
    records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
    record_lines = [
        line
        for line in records_block.splitlines()
        if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
    ]
    selected = [line for line in record_lines if item.lexical_query in line]
    if not selected:
        selected = record_lines[:1]
    candidate_lines = [
        line for line in record_lines if line not in selected and "vault code" in line.lower()
    ]
    if not candidate_lines:
        candidate_lines = [line for line in record_lines if line not in selected]
    prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {selected[0]}\n"
        "Credential descriptor:"
    )
    scored_lines: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + credential_descriptor(line),
        )
        scored_lines.append((score, -line_idx, line))
    if scored_lines:
        scored_lines.sort(reverse=True)
        selected.append(scored_lines[0][2])
    phrase_lm_bridge_prompt = (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join(selected)
        + "\nQuestion:"
        + question
    )
    return tokenizer.encode(phrase_lm_bridge_prompt, add_special_tokens=False)


def compressed_bridge_prompt(
    item: Item,
    question: str,
    query_line: str,
    selected_credential: str,
) -> str:
    source = assigned_descriptor(query_line)
    target = credential_descriptor(selected_credential)
    code = vault_code(selected_credential)
    compressed_lines = [
        f'Field "{item.lexical_query}" means {source}.',
        f"{source} links to {target}.",
    ]
    if code:
        compressed_lines.append(f"{target} code {code}.")
    else:
        compressed_lines.append(selected_credential)
    return (
        "Answer only the three digit value.\n"
        + "\n".join(compressed_lines)
        + "\nQuestion:"
        + question
    )


def compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
) -> list[int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    scored_lines: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + line,
        )
        scored_lines.append((score, -line_idx, line))

    selected_credential = candidate_lines[0] if candidate_lines else ""
    if scored_lines:
        scored_lines.sort(reverse=True)
        selected_credential = scored_lines[0][2]

    return tokenizer.encode(
        compressed_bridge_prompt(item, question, query_line, selected_credential),
        add_special_tokens=False,
    )


def prefilter_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(phrase_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    full_prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    full_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_scored.append((score, -line_idx, line))
    selected = [query_line]
    if full_scored:
        full_scored.sort(reverse=True)
        selected.append(full_scored[0][2])
    prefilter_lm_bridge_prompt = (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join(selected)
        + "\nQuestion:"
        + question
    )
    return (
        tokenizer.encode(prefilter_lm_bridge_prompt, add_special_tokens=False),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(phrase_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    full_prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    full_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if full_scored:
        full_scored.sort(reverse=True)
        selected_credential = full_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def margin_guarded_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    margin_threshold: float = 0.05,
) -> tuple[list[int], int, int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(phrase_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    full_prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    full_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_scored.append((score, -line_idx, line))
    full_scored.sort(reverse=True)

    canonical_prefix = (
        "Match the personnel memo to the normalized credential fact.\n"
        f"Personnel memo: {query_line}\n"
        "Credential fact:"
    )
    canonical_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        descriptor = credential_descriptor(line)
        code = vault_code(line)
        canonical = f" The {descriptor} has vault code {code}."
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            canonical_prefix,
            canonical,
        )
        canonical_scored.append((score, -line_idx, line))
    canonical_scored.sort(reverse=True)

    selected_credential = full_scored[0][2] if full_scored else ""
    override = 0
    if len(full_scored) >= 2 and canonical_scored:
        full_margin = full_scored[0][0] - full_scored[1][0]
        full_runner_up = full_scored[1][2]
        canonical_winner = canonical_scored[0][2]
        if full_margin <= margin_threshold and canonical_winner == full_runner_up:
            selected_credential = canonical_winner
            override = 1

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
        override,
    )


def reverse_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(phrase_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    source_descriptor = assigned_descriptor(query_line)
    reverse_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        reverse_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if reverse_scored:
        reverse_scored.sort(reverse=True)
        selected_credential = reverse_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def bidirectional_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        scored.append((forward_score + reverse_score, -line_idx, line))
    scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(scored)))
    rerank_lines = [line for _score, _line_idx, line in scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    reverse_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        reverse_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if reverse_scored:
        reverse_scored.sort(reverse=True)
        selected_credential = reverse_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        scored.append((forward_score + reverse_score, -line_idx, line))
    scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(scored)))
    rerank_lines = [line for _score, _line_idx, line in scored[:rerank_count]]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    answer_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        code = vault_code(line)
        prompt = compressed_bridge_prompt(item, question, query_line, line) + "\nAnswer:"
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prompt,
            " " + code,
        )
        answer_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if answer_scored:
        answer_scored.sort(reverse=True)
        selected_credential = answer_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def descriptor_pair_match_score(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    candidate_descriptor: str,
) -> float:
    yes_score, no_score = descriptor_pair_match_scores(
        tokenizer,
        model,
        device,
        source_descriptor,
        candidate_descriptor,
    )
    return yes_score - no_score


def descriptor_pair_match_scores(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    candidate_descriptor: str,
) -> tuple[float, float]:
    prefix = (
        "Decide whether two personnel descriptors refer to the same role after "
        "paraphrasing.\n"
        f"Source descriptor: {source_descriptor}\n"
        f"Credential descriptor: {candidate_descriptor}\n"
        "Same role:"
    )
    yes_score = average_continuation_logprob(tokenizer, model, device, prefix, " yes")
    no_score = average_continuation_logprob(tokenizer, model, device, prefix, " no")
    return yes_score, no_score


def lexicon_descriptor_score(
    source_descriptor: str,
    candidate_descriptor: str,
) -> float:
    expected_descriptor = descriptor_phrase_map().get(source_descriptor.lower())
    if expected_descriptor is None:
        return float("-inf")
    expected_tokens = content_tokens(expected_descriptor)
    source_tokens = content_tokens(source_descriptor)
    candidate_tokens = content_tokens(candidate_descriptor)
    expected_hits = len(expected_tokens & candidate_tokens)
    source_leakage = len(source_tokens & candidate_tokens)
    missing_expected = len(expected_tokens - candidate_tokens)
    return float((2 * expected_hits) - source_leakage - missing_expected)


def induced_descriptor_token_map(
    record_lines: list[str],
    query_line: str,
) -> dict[str, set[str]]:
    alias_by_id: dict[int, str] = {}
    profile_by_id: dict[int, str] = {}
    query_id = memo_record_id(query_line)
    for line in record_lines:
        record_id = memo_record_id(line)
        if record_id is None or record_id == query_id:
            continue
        if re.match(r"Memo \d{3}A:", line):
            alias_by_id[record_id] = assigned_descriptor(line).lower()
        elif re.match(r"Memo \d{3}B:", line):
            profile_by_id[record_id] = credential_descriptor(line).lower()

    token_map: dict[str, set[str]] = {}
    for record_id, source_descriptor in alias_by_id.items():
        target_descriptor = profile_by_id.get(record_id)
        if not target_descriptor:
            continue
        source_tokens = [
            token
            for token in source_descriptor.split()
            if token in content_tokens(source_descriptor)
        ]
        target_tokens = [
            token
            for token in target_descriptor.split()
            if token in content_tokens(target_descriptor)
        ]
        for source_token, target_token in zip(source_tokens, target_tokens):
            if source_token != target_token:
                token_map.setdefault(source_token, set()).add(target_token)
    return token_map


def external_support_descriptor_token_map(
    excluded_source_descriptor: str,
) -> dict[str, set[str]]:
    token_map: dict[str, set[str]] = {}
    excluded = excluded_source_descriptor.lower()
    support_pairs = [
        (" ".join(source_terms), " ".join(target_terms))
        for source_terms, target_terms in DESCRIPTOR_FAMILIES
    ]
    support_pairs.extend(HELDOUT_DESCRIPTOR_PAIRS)
    for source_descriptor, target_descriptor in support_pairs:
        if source_descriptor.lower() == excluded:
            continue
        source_tokens = [
            token
            for token in source_descriptor.lower().split()
            if token in content_tokens(source_descriptor)
        ]
        target_tokens = [
            token
            for token in target_descriptor.lower().split()
            if token in content_tokens(target_descriptor)
        ]
        for source_token, target_token in zip(source_tokens, target_tokens):
            if source_token != target_token:
                token_map.setdefault(source_token, set()).add(target_token)
    return token_map


def induced_descriptor_score_parts(
    source_descriptor: str,
    candidate_descriptor: str,
    token_map: dict[str, set[str]],
) -> tuple[float, int, int, int, int]:
    source_tokens = content_tokens(source_descriptor)
    candidate_tokens = content_tokens(candidate_descriptor)
    expected_targets: set[str] = set()
    mapped_source_tokens = 0
    for source_token in source_tokens:
        mapped_targets = token_map.get(source_token, set())
        if mapped_targets:
            mapped_source_tokens += 1
            expected_targets.update(mapped_targets)
    expected_hits = len(expected_targets & candidate_tokens)
    missing_expected = len(expected_targets - candidate_tokens)
    source_leakage = len(source_tokens & candidate_tokens)
    score = float((2 * expected_hits) - missing_expected - source_leakage)
    return score, expected_hits, missing_expected, source_leakage, mapped_source_tokens


def induced_descriptor_score(
    source_descriptor: str,
    candidate_descriptor: str,
    token_map: dict[str, set[str]],
) -> float:
    score, _hits, _missing, _leakage, _mapped = induced_descriptor_score_parts(
        source_descriptor,
        candidate_descriptor,
        token_map,
    )
    return score


def lexicon_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [
        line for _score, _line_idx, line in first_stage_scored[:rerank_count]
    ]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(first_stage_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    lexicon_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = lexicon_descriptor_score(source_descriptor, credential_descriptor(line))
        lexicon_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if lexicon_scored:
        lexicon_scored.sort(reverse=True)
        selected_credential = lexicon_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    token_map = induced_descriptor_token_map(record_lines, query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_scored = first_stage_scored[:rerank_count]
    target_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    induced_scored: list[tuple[float, float, int, str]] = []
    for first_stage_score, neg_line_idx, line in rerank_scored:
        score = induced_descriptor_score(
            source_descriptor,
            credential_descriptor(line),
            token_map,
        )
        induced_scored.append((score, first_stage_score, neg_line_idx, line))
    induced_scored.sort(reverse=True)

    selected_credential = ""
    if induced_scored:
        selected_credential = induced_scored[0][3]
    elif rerank_scored:
        selected_credential = rerank_scored[0][2]

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        rerank_count,
        target_rank,
        target_survived,
    )


def external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    token_map = external_support_descriptor_token_map(source_descriptor)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_scored = first_stage_scored[:rerank_count]
    target_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    induced_scored: list[tuple[float, float, int, str]] = []
    for first_stage_score, neg_line_idx, line in rerank_scored:
        score = induced_descriptor_score(
            source_descriptor,
            credential_descriptor(line),
            token_map,
        )
        induced_scored.append((score, first_stage_score, neg_line_idx, line))
    induced_scored.sort(reverse=True)

    selected_credential = ""
    if induced_scored:
        selected_credential = induced_scored[0][3]
    elif rerank_scored:
        selected_credential = rerank_scored[0][2]

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        rerank_count,
        target_rank,
        target_survived,
    )


def strict_induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    token_map = induced_descriptor_token_map(record_lines, query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_scored = first_stage_scored[:rerank_count]
    target_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    induced_scored: list[tuple[float, int, str]] = []
    for _first_stage_score, neg_line_idx, line in rerank_scored:
        score = induced_descriptor_score(
            source_descriptor,
            credential_descriptor(line),
            token_map,
        )
        induced_scored.append((score, neg_line_idx, line))
    induced_scored.sort(reverse=True)

    selected_credential = ""
    if induced_scored:
        selected_credential = induced_scored[0][2]
    elif rerank_scored:
        selected_credential = rerank_scored[0][2]

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        rerank_count,
        target_rank,
        target_survived,
    )


def fewshot_descriptor_translation_prefix(source_descriptor: str) -> str:
    examples = [
        ("coastal route analyst", "nautical chart planner"),
        ("glass archive reviewer", "crystal record curator"),
        ("copper garden inspector", "bronze plant examiner"),
        ("radio tower builder", "signal mast constructor"),
        ("winter market broker", "frost exchange agent"),
        ("quiet flight operator", "silent aircraft pilot"),
        ("river cargo scheduler", "stream freight planner"),
        ("urban bridge designer", "city span architect"),
    ]
    lines = [
        "Translate each source-side personnel descriptor into its target-side credential descriptor.",
    ]
    for source, target in examples:
        lines.extend(
            [
                f"Source descriptor: {source}",
                f"Target descriptor: {target}",
            ]
        )
    lines.extend(
        [
            f"Source descriptor: {source_descriptor}",
            "Target descriptor:",
        ]
    )
    return "\n".join(lines)


def fewshot_descriptor_translation_score(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    candidate_descriptor: str,
) -> float:
    score, _translation_score, _leakage_penalty = fewshot_descriptor_translation_score_parts(
        tokenizer,
        model,
        device,
        source_descriptor,
        candidate_descriptor,
    )
    return score


def fewshot_descriptor_translation_score_parts(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    candidate_descriptor: str,
) -> tuple[float, float, float]:
    translation_score = average_continuation_logprob(
        tokenizer,
        model,
        device,
        fewshot_descriptor_translation_prefix(source_descriptor),
        " " + candidate_descriptor,
    )
    source_leakage = len(content_tokens(source_descriptor) & content_tokens(candidate_descriptor))
    leakage_penalty = 0.25 * source_leakage
    return translation_score - leakage_penalty, translation_score, leakage_penalty


def fewshot_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [
        line for _score, _line_idx, line in first_stage_scored[:rerank_count]
    ]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(first_stage_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    translation_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = fewshot_descriptor_translation_score(
            tokenizer,
            model,
            device,
            source_descriptor,
            credential_descriptor(line),
        )
        translation_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if translation_scored:
        translation_scored.sort(reverse=True)
        selected_credential = translation_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def forced_choice_descriptor_score(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    labeled_descriptors: list[tuple[str, str]],
    label: str,
) -> float:
    lines = [
        "Choose the one credential descriptor that best matches the source descriptor after paraphrasing.",
        f"Source descriptor: {source_descriptor}",
        "Candidate credential descriptors:",
    ]
    for candidate_label, candidate_descriptor in labeled_descriptors:
        lines.append(f"{candidate_label}. {candidate_descriptor}")
    lines.append("Best candidate:")
    return average_continuation_logprob(
        tokenizer,
        model,
        device,
        "\n".join(lines),
        " " + label,
    )


def forced_choice_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [
        line for _score, _line_idx, line in first_stage_scored[:rerank_count]
    ]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(first_stage_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    labels = [chr(ord("A") + index) for index in range(len(rerank_lines))]
    labeled_descriptors = [
        (label, credential_descriptor(line))
        for label, line in zip(labels, rerank_lines)
    ]
    choice_scored: list[tuple[float, int, str]] = []
    for line_idx, (label, line) in enumerate(zip(labels, rerank_lines)):
        score = forced_choice_descriptor_score(
            tokenizer,
            model,
            device,
            source_descriptor,
            labeled_descriptors,
            label,
        )
        choice_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if choice_scored:
        choice_scored.sort(reverse=True)
        selected_credential = choice_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def pairwise_descriptor_preference_margin(
    tokenizer,
    model,
    device: torch.device,
    source_descriptor: str,
    left_descriptor: str,
    right_descriptor: str,
) -> float:
    def label_margin(
        first_descriptor: str,
        second_descriptor: str,
        preferred_label: str,
        other_label: str,
    ) -> float:
        prefix = (
            "Choose which credential descriptor best paraphrases the source descriptor.\n"
            f"Source descriptor: {source_descriptor}\n"
            f"A. {first_descriptor}\n"
            f"B. {second_descriptor}\n"
            "Best candidate:"
        )
        preferred_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + preferred_label,
        )
        other_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + other_label,
        )
        return preferred_score - other_score

    left_as_a = label_margin(left_descriptor, right_descriptor, "A", "B")
    left_as_b = label_margin(right_descriptor, left_descriptor, "B", "A")
    return 0.5 * (left_as_a + left_as_b)


def pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [
        line for _score, _line_idx, line in first_stage_scored[:rerank_count]
    ]
    target_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    selected_credential = rerank_lines[0] if rerank_lines else ""
    for challenger in rerank_lines[1:]:
        margin = pairwise_descriptor_preference_margin(
            tokenizer,
            model,
            device,
            source_descriptor,
            credential_descriptor(challenger),
            credential_descriptor(selected_credential),
        )
        if margin > 0.0:
            selected_credential = challenger

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    source_tokens = content_tokens(source_descriptor)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_scored = first_stage_scored[:rerank_count]
    target_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    guarded_scored: list[tuple[float, float, int, str]] = []
    for first_stage_score, neg_line_idx, line in rerank_scored:
        source_overlap = len(source_tokens & content_tokens(credential_descriptor(line)))
        guarded_score = first_stage_score - (2.0 * source_overlap)
        guarded_scored.append((guarded_score, first_stage_score, neg_line_idx, line))
    guarded_scored.sort(reverse=True)

    selected_credential = ""
    if guarded_scored:
        selected_credential = guarded_scored[0][3]
    elif rerank_scored:
        selected_credential = rerank_scored[0][2]

    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        rerank_count,
        target_rank,
        target_survived,
    )


def contrastive_descriptor_prefilter_compressed_lm_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
) -> tuple[list[int], int, int, int, int]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [
        line for _score, _line_idx, line in first_stage_scored[:rerank_count]
    ]
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(first_stage_scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    target_survived = int(target_rank != 0 and target_rank <= rerank_count)

    verifier_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        score = descriptor_pair_match_score(
            tokenizer,
            model,
            device,
            source_descriptor,
            credential_descriptor(line),
        )
        verifier_scored.append((score, -line_idx, line))

    selected_credential = rerank_lines[0] if rerank_lines else ""
    if verifier_scored:
        verifier_scored.sort(reverse=True)
        selected_credential = verifier_scored[0][2]
    return (
        tokenizer.encode(
            compressed_bridge_prompt(item, question, query_line, selected_credential),
            add_special_tokens=False,
        ),
        len(candidate_lines),
        len(rerank_lines),
        target_rank,
        target_survived,
    )


def rank_of_line(scored: list[tuple[float, int, str]], target_line: str) -> int:
    for rank, (_score, _line_idx, line) in enumerate(scored, start=1):
        if line == target_line:
            return rank
    return 0


def diagnostic_score_entry(
    score: float,
    line: str,
    target_line: str,
) -> dict[str, float | int | str]:
    return {
        "score": round(score, 6),
        "is_target": int(line == target_line),
        "descriptor": credential_descriptor(line),
        "code": vault_code(line),
        "line": line,
    }


def bidirectional_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
) -> tuple[dict[str, float | int], dict[str, object]]:
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    forward_scores: dict[str, float] = {}
    reverse_scores: dict[str, float] = {}
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        forward_scores[line] = forward_score
        reverse_scores[line] = reverse_score
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]

    reverse_final_scored = [
        (reverse_scores[line], -line_idx, line) for line_idx, line in enumerate(rerank_lines)
    ]
    reverse_final_scored.sort(reverse=True)

    full_prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    full_memo_scored: list[tuple[float, int, str]] = []
    qa_answer_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(rerank_lines):
        full_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_memo_scored.append((full_score, -line_idx, line))

        code = vault_code(line)
        answer_prompt = compressed_bridge_prompt(item, question, query_line, line) + "\nAnswer:"
        qa_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            answer_prompt,
            " " + code,
        )
        qa_answer_scored.append((qa_score, -line_idx, line))
    full_memo_scored.sort(reverse=True)
    qa_answer_scored.sort(reverse=True)

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    reverse_final_rank = rank_of_line(reverse_final_scored, item.target_record)
    full_memo_rank = rank_of_line(full_memo_scored, item.target_record)
    qa_answer_rank = rank_of_line(qa_answer_scored, item.target_record)

    summary: dict[str, float | int] = {
        "diagnostic_candidate_count": len(candidate_lines),
        "diagnostic_rerank_count": rerank_count,
        "diagnostic_first_stage_target_rank": first_stage_rank,
        "diagnostic_target_survived": target_survived,
        "diagnostic_reverse_final_target_rank": reverse_final_rank,
        "diagnostic_full_memo_final_target_rank": full_memo_rank,
        "diagnostic_qa_answer_final_target_rank": qa_answer_rank,
        "diagnostic_reverse_final_selection_accuracy": int(
            bool(reverse_final_scored) and reverse_final_scored[0][2] == item.target_record
        ),
        "diagnostic_full_memo_final_selection_accuracy": int(
            bool(full_memo_scored) and full_memo_scored[0][2] == item.target_record
        ),
        "diagnostic_qa_answer_final_selection_accuracy": int(
            bool(qa_answer_scored) and qa_answer_scored[0][2] == item.target_record
        ),
    }

    trace_top_k = max(1, top_k)
    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[:trace_top_k]
        ],
        "reverse_final_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in reverse_final_scored[:trace_top_k]
        ],
        "full_memo_final_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in full_memo_scored[:trace_top_k]
        ],
        "qa_answer_final_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in qa_answer_scored[:trace_top_k]
        ],
    }
    return summary, trace


def contrastive_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
) -> tuple[dict[str, float | int], dict[str, object]]:
    _question, _record_lines, query_line, candidate_lines = query_and_credential_lines(
        item
    )
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]

    verifier_scored: list[tuple[float, int, str]] = []
    verifier_details: dict[str, tuple[float, float, float]] = {}
    for line_idx, line in enumerate(rerank_lines):
        yes_score, no_score = descriptor_pair_match_scores(
            tokenizer,
            model,
            device,
            source_descriptor,
            credential_descriptor(line),
        )
        margin = yes_score - no_score
        verifier_details[line] = (yes_score, no_score, margin)
        verifier_scored.append((margin, -line_idx, line))
    verifier_scored.sort(reverse=True)

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    verifier_rank = rank_of_line(verifier_scored, item.target_record)
    selected_line = verifier_scored[0][2] if verifier_scored else ""
    target_margin = (
        verifier_details[item.target_record][2]
        if item.target_record in verifier_details
        else 0.0
    )
    selected_margin = verifier_details[selected_line][2] if selected_line else 0.0
    non_target_margins = [
        margin for _yes, _no, margin in verifier_details.values()
    ]
    if item.target_record in verifier_details:
        non_target_margins = [
            margin
            for line, (_yes, _no, margin) in verifier_details.items()
            if line != item.target_record
        ]
    best_non_target_margin = max(non_target_margins) if non_target_margins else 0.0
    target_gap_vs_best_non_target = (
        target_margin - best_non_target_margin
        if item.target_record in verifier_details and non_target_margins
        else 0.0
    )

    summary: dict[str, float | int] = {
        "contrastive_diagnostic_candidate_count": len(candidate_lines),
        "contrastive_diagnostic_rerank_count": rerank_count,
        "contrastive_diagnostic_first_stage_target_rank": first_stage_rank,
        "contrastive_diagnostic_target_survived": target_survived,
        "contrastive_diagnostic_verifier_target_rank": verifier_rank,
        "contrastive_diagnostic_verifier_selection_accuracy": int(
            bool(verifier_scored) and selected_line == item.target_record
        ),
        "contrastive_diagnostic_target_margin": target_margin,
        "contrastive_diagnostic_selected_margin": selected_margin,
        "contrastive_diagnostic_best_non_target_margin": best_non_target_margin,
        "contrastive_diagnostic_target_gap_vs_best_non_target": target_gap_vs_best_non_target,
    }

    trace_top_k = max(1, top_k)
    verifier_top: list[dict[str, float | int | str]] = []
    for margin, _line_idx, line in verifier_scored[:trace_top_k]:
        yes_score, no_score, _margin = verifier_details[line]
        entry = diagnostic_score_entry(margin, line, item.target_record)
        entry["yes_score"] = round(yes_score, 6)
        entry["no_score"] = round(no_score, 6)
        verifier_top.append(entry)

    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[:trace_top_k]
        ],
        "contrastive_verifier_top": verifier_top,
    }
    return summary, trace


def fewshot_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
) -> tuple[dict[str, float | int], dict[str, object]]:
    _question, _record_lines, query_line, candidate_lines = query_and_credential_lines(
        item
    )
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]

    fewshot_scored: list[tuple[float, int, str]] = []
    fewshot_details: dict[str, tuple[float, float, float]] = {}
    for line_idx, line in enumerate(rerank_lines):
        score, translation_score, leakage_penalty = (
            fewshot_descriptor_translation_score_parts(
                tokenizer,
                model,
                device,
                source_descriptor,
                credential_descriptor(line),
            )
        )
        fewshot_details[line] = (score, translation_score, leakage_penalty)
        fewshot_scored.append((score, -line_idx, line))
    fewshot_scored.sort(reverse=True)

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    fewshot_rank = rank_of_line(fewshot_scored, item.target_record)
    selected_line = fewshot_scored[0][2] if fewshot_scored else ""
    target_score = (
        fewshot_details[item.target_record][0]
        if item.target_record in fewshot_details
        else 0.0
    )
    selected_score = fewshot_details[selected_line][0] if selected_line else 0.0
    non_target_scores = [
        score
        for line, (score, _translation_score, _leakage_penalty) in fewshot_details.items()
        if line != item.target_record
    ]
    best_non_target_score = max(non_target_scores) if non_target_scores else 0.0
    target_gap_vs_best_non_target = (
        target_score - best_non_target_score
        if item.target_record in fewshot_details and non_target_scores
        else 0.0
    )
    target_leakage_penalty = (
        fewshot_details[item.target_record][2]
        if item.target_record in fewshot_details
        else 0.0
    )

    summary: dict[str, float | int] = {
        "fewshot_diagnostic_candidate_count": len(candidate_lines),
        "fewshot_diagnostic_rerank_count": rerank_count,
        "fewshot_diagnostic_first_stage_target_rank": first_stage_rank,
        "fewshot_diagnostic_target_survived": target_survived,
        "fewshot_diagnostic_final_target_rank": fewshot_rank,
        "fewshot_diagnostic_selection_accuracy": int(
            bool(fewshot_scored) and selected_line == item.target_record
        ),
        "fewshot_diagnostic_target_score": target_score,
        "fewshot_diagnostic_selected_score": selected_score,
        "fewshot_diagnostic_best_non_target_score": best_non_target_score,
        "fewshot_diagnostic_target_gap_vs_best_non_target": target_gap_vs_best_non_target,
        "fewshot_diagnostic_target_leakage_penalty": target_leakage_penalty,
    }

    trace_top_k = max(1, top_k)
    fewshot_top: list[dict[str, float | int | str]] = []
    for score, _line_idx, line in fewshot_scored[:trace_top_k]:
        _score, translation_score, leakage_penalty = fewshot_details[line]
        entry = diagnostic_score_entry(score, line, item.target_record)
        entry["translation_score"] = round(translation_score, 6)
        entry["leakage_penalty"] = round(leakage_penalty, 6)
        fewshot_top.append(entry)

    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[:trace_top_k]
        ],
        "fewshot_translation_top": fewshot_top,
    }
    return summary, trace


def forced_choice_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
) -> tuple[dict[str, float | int], dict[str, object]]:
    _question, _record_lines, query_line, candidate_lines = query_and_credential_lines(
        item
    )
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]
    labels = [chr(ord("A") + index) for index in range(len(rerank_lines))]
    labeled_descriptors = [
        (label, credential_descriptor(line))
        for label, line in zip(labels, rerank_lines)
    ]

    choice_scored: list[tuple[float, int, str, str]] = []
    choice_details: dict[str, tuple[float, str]] = {}
    for line_idx, (label, line) in enumerate(zip(labels, rerank_lines)):
        score = forced_choice_descriptor_score(
            tokenizer,
            model,
            device,
            source_descriptor,
            labeled_descriptors,
            label,
        )
        choice_details[line] = (score, label)
        choice_scored.append((score, -line_idx, line, label))
    choice_scored.sort(reverse=True)

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    target_rank = rank_of_line(
        [(score, line_idx, line) for score, line_idx, line, _label in choice_scored],
        item.target_record,
    )
    selected_line = choice_scored[0][2] if choice_scored else ""
    selected_score = choice_details[selected_line][0] if selected_line else 0.0
    target_score = (
        choice_details[item.target_record][0]
        if item.target_record in choice_details
        else 0.0
    )
    non_target_scores = [
        score
        for line, (score, _label) in choice_details.items()
        if line != item.target_record
    ]
    best_non_target_score = max(non_target_scores) if non_target_scores else 0.0
    target_gap_vs_best_non_target = (
        target_score - best_non_target_score
        if item.target_record in choice_details and non_target_scores
        else 0.0
    )
    target_label_ord = (
        float(ord(choice_details[item.target_record][1]) - ord("A"))
        if item.target_record in choice_details
        else -1.0
    )
    selected_label_ord = (
        float(ord(choice_details[selected_line][1]) - ord("A"))
        if selected_line
        else -1.0
    )

    summary: dict[str, float | int] = {
        "forced_choice_diagnostic_candidate_count": len(candidate_lines),
        "forced_choice_diagnostic_rerank_count": rerank_count,
        "forced_choice_diagnostic_first_stage_target_rank": first_stage_rank,
        "forced_choice_diagnostic_target_survived": target_survived,
        "forced_choice_diagnostic_final_target_rank": target_rank,
        "forced_choice_diagnostic_selection_accuracy": int(
            bool(choice_scored) and selected_line == item.target_record
        ),
        "forced_choice_diagnostic_target_score": target_score,
        "forced_choice_diagnostic_selected_score": selected_score,
        "forced_choice_diagnostic_best_non_target_score": best_non_target_score,
        "forced_choice_diagnostic_target_gap_vs_best_non_target": target_gap_vs_best_non_target,
        "forced_choice_diagnostic_target_label_ord": target_label_ord,
        "forced_choice_diagnostic_selected_label_ord": selected_label_ord,
    }

    trace_top_k = max(1, top_k)
    choice_top: list[dict[str, float | int | str]] = []
    for score, _line_idx, line, label in choice_scored[:trace_top_k]:
        entry = diagnostic_score_entry(score, line, item.target_record)
        entry["label"] = label
        entry["label_ord"] = ord(label) - ord("A")
        choice_top.append(entry)

    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[:trace_top_k]
        ],
        "forced_choice_top": choice_top,
    }
    return summary, trace


def pairwise_tournament_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
) -> tuple[dict[str, float | int], dict[str, object]]:
    _question, _record_lines, query_line, candidate_lines = query_and_credential_lines(
        item
    )
    source_descriptor = assigned_descriptor(query_line)
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]
    selected_line = rerank_lines[0] if rerank_lines else ""
    selected_initial_rank = 1 if selected_line else 0
    target_entered_step = 0
    target_won_step = 0
    target_lost_step = 0
    comparisons: list[dict[str, float | int | str]] = []

    for step_idx, challenger in enumerate(rerank_lines[1:], start=2):
        incumbent = selected_line
        margin = pairwise_descriptor_preference_margin(
            tokenizer,
            model,
            device,
            source_descriptor,
            credential_descriptor(challenger),
            credential_descriptor(incumbent),
        )
        challenger_wins = margin > 0.0
        if challenger == item.target_record:
            target_entered_step = step_idx
            if challenger_wins:
                target_won_step = step_idx
            else:
                target_lost_step = step_idx
        elif incumbent == item.target_record and challenger_wins:
            target_lost_step = step_idx
        if len(comparisons) < max(1, top_k):
            comparisons.append(
                {
                    "step": step_idx,
                    "margin_challenger_vs_incumbent": round(margin, 6),
                    "challenger_wins": int(challenger_wins),
                    "incumbent_descriptor": credential_descriptor(incumbent),
                    "incumbent_is_target": int(incumbent == item.target_record),
                    "challenger_descriptor": credential_descriptor(challenger),
                    "challenger_is_target": int(challenger == item.target_record),
                }
            )
        if challenger_wins:
            selected_line = challenger
            selected_initial_rank = step_idx

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    summary: dict[str, float | int] = {
        "pairwise_tournament_diagnostic_candidate_count": len(candidate_lines),
        "pairwise_tournament_diagnostic_rerank_count": rerank_count,
        "pairwise_tournament_diagnostic_first_stage_target_rank": first_stage_rank,
        "pairwise_tournament_diagnostic_target_survived": target_survived,
        "pairwise_tournament_diagnostic_selection_accuracy": int(
            selected_line == item.target_record
        ),
        "pairwise_tournament_diagnostic_selected_initial_rank": selected_initial_rank,
        "pairwise_tournament_diagnostic_target_entered_step": target_entered_step,
        "pairwise_tournament_diagnostic_target_won_step": target_won_step,
        "pairwise_tournament_diagnostic_target_lost_step": target_lost_step,
    }

    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "selected_descriptor": credential_descriptor(selected_line) if selected_line else "",
        "selected_is_target": int(selected_line == item.target_record),
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[: max(1, top_k)]
        ],
        "tournament_comparisons": comparisons,
    }
    return summary, trace


def induced_map_final_score_diagnostic(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
    prefilter_candidates: int,
    top_k: int,
    map_source: str = "in_context",
) -> tuple[dict[str, float | int], dict[str, object]]:
    _question, record_lines, query_line, candidate_lines = query_and_credential_lines(
        item
    )
    source_descriptor = assigned_descriptor(query_line)
    if map_source == "external_support":
        token_map = external_support_descriptor_token_map(source_descriptor)
        metric_prefix = "external_support_induced_map_diagnostic"
    elif map_source == "in_context":
        token_map = induced_descriptor_token_map(record_lines, query_line)
        metric_prefix = "induced_map_diagnostic"
    else:
        raise ValueError(f"unknown induced-map diagnostic source: {map_source}")
    forward_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )

    first_stage_scored: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        forward_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            forward_prefix,
            " " + credential_descriptor(line),
        )
        reverse_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        reverse_score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            reverse_prefix,
            " " + source_descriptor,
        )
        first_stage_scored.append((forward_score + reverse_score, -line_idx, line))
    first_stage_scored.sort(reverse=True)

    rerank_count = max(1, min(prefilter_candidates, len(first_stage_scored)))
    rerank_lines = [line for _score, _line_idx, line in first_stage_scored[:rerank_count]]

    induced_scored: list[tuple[float, int, str]] = []
    induced_details: dict[str, tuple[float, int, int, int, int]] = {}
    for line_idx, line in enumerate(rerank_lines):
        parts = induced_descriptor_score_parts(
            source_descriptor,
            credential_descriptor(line),
            token_map,
        )
        induced_details[line] = parts
        induced_scored.append((parts[0], -line_idx, line))
    induced_scored.sort(reverse=True)

    first_stage_rank = rank_of_line(first_stage_scored, item.target_record)
    first_stage_selected_line = first_stage_scored[0][2] if first_stage_scored else ""
    first_stage_target_score = 0.0
    first_stage_best_non_target_score = 0.0
    first_stage_non_target_scores: list[float] = []
    for score, _line_idx, line in first_stage_scored:
        if line == item.target_record:
            first_stage_target_score = score
        else:
            first_stage_non_target_scores.append(score)
    if first_stage_non_target_scores:
        first_stage_best_non_target_score = max(first_stage_non_target_scores)
    first_stage_target_gap_vs_best_non_target = (
        first_stage_target_score - first_stage_best_non_target_score
        if first_stage_rank != 0 and first_stage_non_target_scores
        else 0.0
    )
    target_survived = int(first_stage_rank != 0 and first_stage_rank <= rerank_count)
    final_rank = rank_of_line(induced_scored, item.target_record)
    selected_line = induced_scored[0][2] if induced_scored else ""
    first_stage_selection_accuracy = int(
        bool(first_stage_selected_line)
        and first_stage_selected_line == item.target_record
    )
    final_selection_accuracy = int(bool(induced_scored) and selected_line == item.target_record)
    target_score = (
        induced_details[item.target_record][0]
        if item.target_record in induced_details
        else 0.0
    )
    selected_score = induced_details[selected_line][0] if selected_line else 0.0
    non_target_scores = [
        score
        for line, (score, _hits, _missing, _leakage, _mapped) in induced_details.items()
        if line != item.target_record
    ]
    best_non_target_score = max(non_target_scores) if non_target_scores else 0.0
    target_gap_vs_best_non_target = (
        target_score - best_non_target_score
        if item.target_record in induced_details and non_target_scores
        else 0.0
    )
    source_tokens = content_tokens(source_descriptor)
    mapped_source_tokens = sum(1 for token in source_tokens if token_map.get(token))
    coverage = mapped_source_tokens / max(1, len(source_tokens))
    zero_coverage = int(mapped_source_tokens == 0)
    final_changed_first_stage = int(
        bool(selected_line)
        and bool(first_stage_selected_line)
        and selected_line != first_stage_selected_line
    )
    positive_target_gap = int(target_gap_vs_best_non_target > 0.0)

    summary: dict[str, float | int] = {
        f"{metric_prefix}_candidate_count": len(candidate_lines),
        f"{metric_prefix}_rerank_count": rerank_count,
        f"{metric_prefix}_first_stage_target_rank": first_stage_rank,
        f"{metric_prefix}_first_stage_selection_accuracy": first_stage_selection_accuracy,
        f"{metric_prefix}_first_stage_target_score": first_stage_target_score,
        f"{metric_prefix}_first_stage_selected_score": (
            first_stage_scored[0][0] if first_stage_scored else 0.0
        ),
        f"{metric_prefix}_first_stage_best_non_target_score": first_stage_best_non_target_score,
        f"{metric_prefix}_first_stage_target_gap_vs_best_non_target": first_stage_target_gap_vs_best_non_target,
        f"{metric_prefix}_target_survived": target_survived,
        f"{metric_prefix}_final_target_rank": final_rank,
        f"{metric_prefix}_selection_accuracy": final_selection_accuracy,
        f"{metric_prefix}_target_score": target_score,
        f"{metric_prefix}_selected_score": selected_score,
        f"{metric_prefix}_best_non_target_score": best_non_target_score,
        f"{metric_prefix}_target_gap_vs_best_non_target": target_gap_vs_best_non_target,
        f"{metric_prefix}_source_token_coverage": coverage,
        f"{metric_prefix}_zero_coverage": zero_coverage,
        f"{metric_prefix}_zero_coverage_selection_success": (
            zero_coverage * final_selection_accuracy
        ),
        f"{metric_prefix}_final_changed_first_stage_selection": final_changed_first_stage,
        f"{metric_prefix}_positive_target_gap": positive_target_gap,
    }

    trace_top_k = max(1, top_k)
    induced_top: list[dict[str, float | int | str]] = []
    for score, _line_idx, line in induced_scored[:trace_top_k]:
        _score, hits, missing, leakage, mapped = induced_details[line]
        entry = diagnostic_score_entry(score, line, item.target_record)
        entry["expected_hits"] = hits
        entry["missing_expected"] = missing
        entry["source_leakage"] = leakage
        entry["mapped_source_tokens"] = mapped
        induced_top.append(entry)

    trace: dict[str, object] = {
        "query_line": query_line,
        "source_descriptor": source_descriptor,
        "target_descriptor": credential_descriptor(item.target_record),
        "target_code": item.answer,
        "map_source": map_source,
        "token_map": {key: sorted(value) for key, value in sorted(token_map.items())},
        "summary": summary,
        "first_stage_top": [
            diagnostic_score_entry(score, line, item.target_record)
            for score, _line_idx, line in first_stage_scored[:trace_top_k]
        ],
        "induced_map_top": induced_top,
    }
    return summary, trace


@torch.inference_mode()
def mean_hidden_embedding(
    tokenizer,
    model,
    device: torch.device,
    text: str,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("cannot embed empty text")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0]
    return torch.nn.functional.normalize(hidden.mean(dim=0), dim=0)


def embedding_bridge_token_ids(
    tokenizer,
    model,
    device: torch.device,
    item: Item,
) -> list[int]:
    records_block, question = item.prompt.rsplit("\nQuestion:", maxsplit=1)
    record_lines = [
        line
        for line in records_block.splitlines()
        if line.startswith(("Record ", "Alias ", "Profile ", "Memo "))
    ]
    selected = [line for line in record_lines if item.lexical_query in line]
    if not selected:
        selected = record_lines[:1]
    candidate_lines = [
        line for line in record_lines if line not in selected and "vault code" in line.lower()
    ]
    if not candidate_lines:
        candidate_lines = [line for line in record_lines if line not in selected]
    query_embedding = mean_hidden_embedding(tokenizer, model, device, selected[0])
    scored_lines: list[tuple[float, int, str]] = []
    for line_idx, line in enumerate(candidate_lines):
        line_embedding = mean_hidden_embedding(tokenizer, model, device, line)
        score = float(torch.dot(query_embedding, line_embedding).item())
        scored_lines.append((score, -line_idx, line))
    if scored_lines:
        scored_lines.sort(reverse=True)
        selected.append(scored_lines[0][2])
    embedding_bridge_prompt = (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join(selected)
        + "\nQuestion:"
        + question
    )
    return tokenizer.encode(embedding_bridge_prompt, add_special_tokens=False)


CONTENT_STOPWORDS = {
    "a",
    "about",
    "answer",
    "as",
    "assigned",
    "audit",
    "auditors",
    "badge",
    "behind",
    "code",
    "credential",
    "dispatch",
    "during",
    "employee",
    "field",
    "file",
    "for",
    "gave",
    "given",
    "id",
    "in",
    "is",
    "lists",
    "maps",
    "memo",
    "notes",
    "number",
    "onboarding",
    "only",
    "person",
    "points",
    "question",
    "recorded",
    "records",
    "reviewers",
    "security",
    "setup",
    "staff",
    "the",
    "title",
    "to",
    "toward",
    "use",
    "vault",
    "was",
    "when",
    "with",
    "writing",
}


def normalize_identifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def extract_identifiers(text: str) -> set[str]:
    return {
        normalize_identifier(match.group(0))
        for match in re.finditer(r"\b[A-Z]{2}-?\d{3}\b", text)
    }


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
        if token not in CONTENT_STOPWORDS and len(token) > 1
    }


def semantic_equivalents() -> dict[str, set[str]]:
    equivalents: dict[str, set[str]] = {}
    for source_terms, target_terms in DESCRIPTOR_FAMILIES:
        cluster = set(source_terms) | set(target_terms)
        for term in cluster:
            equivalents.setdefault(term, set()).update(cluster)
    return equivalents


SEMANTIC_EQUIVALENTS = semantic_equivalents()


def descriptor_phrase_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx in range(len(DESCRIPTOR_FAMILIES) * 6):
        source, target = descriptor_pair(idx)
        mapping[source.lower()] = target.lower()
    for source, target in HELDOUT_DESCRIPTOR_PAIRS:
        mapping[source.lower()] = target.lower()
    return mapping


def semantic_tokens(text: str) -> set[str]:
    tokens = content_tokens(text)
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SEMANTIC_EQUIVALENTS.get(token, ()))
    return expanded


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_identifier(text)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[idx : idx + n] for idx in range(len(normalized) - n + 1)}


def bridge_char_grams(text: str, n: int = 3) -> set[str]:
    raw_units = re.findall(r"[A-Za-z0-9]+", text)
    normalized_units = [normalize_identifier(unit) for unit in raw_units]
    normalized_units = [unit for unit in normalized_units if unit]
    stopword_units = {normalize_identifier(word) for word in CONTENT_STOPWORDS}
    units = [
        unit
        for unit in normalized_units
        if unit not in stopword_units and not re.fullmatch(r"\d{3}[A-Z]?", unit)
    ]
    pair_units: list[str] = []
    for left, right in zip(normalized_units, normalized_units[1:]):
        if (
            (left.isalpha() and right.isdigit() and len(left) <= 3)
            or (left.isdigit() and right.isalpha() and len(right) <= 3)
        ):
            pair_units.append(left + right)
    grams: set[str] = set()
    for unit in units + pair_units:
        grams.update(char_ngrams(unit, n=n))
    return grams


def char_ngram_idf(lines: list[str]) -> dict[str, float]:
    document_frequency: dict[str, int] = {}
    for line in lines:
        for gram in bridge_char_grams(line):
            document_frequency[gram] = document_frequency.get(gram, 0) + 1
    total = len(lines)
    return {
        gram: np.log((1 + total) / (1 + count)) + 1.0
        for gram, count in document_frequency.items()
    }


def first_int(text: str) -> str | None:
    match = re.search(r"\b\d{3}\b", text)
    return match.group(0) if match else None


@torch.inference_mode()
def answer_one(
    tokenizer,
    model,
    device: torch.device,
    token_ids: list[int],
    max_new_tokens: int,
) -> str:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def rank_answer(
    tokenizer,
    model,
    device: torch.device,
    token_ids: list[int],
    candidates: tuple[str, ...],
) -> str:
    encoded_candidates = [
        (candidate, tokenizer.encode(candidate, add_special_tokens=False))
        for candidate in candidates
    ]
    encoded_candidates = [
        (candidate, candidate_ids)
        for candidate, candidate_ids in encoded_candidates
        if candidate_ids
    ]
    if not encoded_candidates:
        raise RuntimeError("candidate ranking produced no scores")

    prompt_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prompt_outputs = model(input_ids=prompt_ids, use_cache=True)
    first_token_log_probs = torch.log_softmax(prompt_outputs.logits[0, -1], dim=-1)
    scored: list[tuple[float, str]] = []
    for candidate, candidate_ids in encoded_candidates:
        total = float(first_token_log_probs[candidate_ids[0]].item())
        if len(candidate_ids) > 1:
            continuation_ids = torch.tensor(
                [candidate_ids[:-1]],
                dtype=torch.long,
                device=device,
            )
            continuation_outputs = model(
                input_ids=continuation_ids,
                past_key_values=prompt_outputs.past_key_values,
                use_cache=False,
            )
            continuation_log_probs = torch.log_softmax(
                continuation_outputs.logits[0],
                dim=-1,
            )
            for offset, candidate_id in enumerate(candidate_ids[1:]):
                total += float(
                    continuation_log_probs[offset, candidate_id].item()
                )
        scored.append((total / len(candidate_ids), candidate))
    return max(scored)[1]


def evaluate_with_model(
    args: argparse.Namespace,
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, float | int | str]:
    seed_everything(args.seed)
    items = build_task_items(
        args.task,
        args.items,
        args.distractors,
        args.seed,
        args.target_position,
    )
    policies = [
        "full",
        "truncate",
        "streaming",
        "oracle_span",
        "lexical",
        "bridge",
        "prose_bridge",
        "entity_bridge",
        "overlap_bridge",
        "id_bridge",
        "char_bridge",
        "neighbor_bridge",
        "semantic_bridge",
        "embedding_bridge",
        "phrase_lm_bridge",
        "prefilter_lm_bridge",
        "prefilter_compressed_lm_bridge",
        "margin_guarded_prefilter_compressed_lm_bridge",
        "reverse_descriptor_prefilter_compressed_lm_bridge",
        "bidirectional_descriptor_prefilter_compressed_lm_bridge",
        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge",
        "lexicon_descriptor_prefilter_compressed_lm_bridge",
        "induced_map_descriptor_prefilter_compressed_lm_bridge",
        "external_support_induced_map_descriptor_prefilter_compressed_lm_bridge",
        "strict_induced_map_descriptor_prefilter_compressed_lm_bridge",
        "fewshot_descriptor_prefilter_compressed_lm_bridge",
        "forced_choice_descriptor_prefilter_compressed_lm_bridge",
        "pairwise_tournament_descriptor_prefilter_compressed_lm_bridge",
        "source_copy_guarded_descriptor_prefilter_compressed_lm_bridge",
        "contrastive_descriptor_prefilter_compressed_lm_bridge",
        "compressed_lm_bridge",
        "lm_bridge",
    ]
    correct = {policy: 0 for policy in policies}
    kept_fracs = {policy: [] for policy in policies}
    candidate_counts: list[int] = []
    rerank_counts: list[int] = []
    target_ranks: list[int] = []
    target_survivals: list[int] = []
    compressed_candidate_counts: list[int] = []
    compressed_rerank_counts: list[int] = []
    compressed_target_ranks: list[int] = []
    compressed_target_survivals: list[int] = []
    guarded_candidate_counts: list[int] = []
    guarded_rerank_counts: list[int] = []
    guarded_target_ranks: list[int] = []
    guarded_target_survivals: list[int] = []
    guarded_overrides: list[int] = []
    reverse_descriptor_candidate_counts: list[int] = []
    reverse_descriptor_rerank_counts: list[int] = []
    reverse_descriptor_target_ranks: list[int] = []
    reverse_descriptor_target_survivals: list[int] = []
    bidirectional_descriptor_candidate_counts: list[int] = []
    bidirectional_descriptor_rerank_counts: list[int] = []
    bidirectional_descriptor_target_ranks: list[int] = []
    bidirectional_descriptor_target_survivals: list[int] = []
    bidirectional_descriptor_qa_candidate_counts: list[int] = []
    bidirectional_descriptor_qa_rerank_counts: list[int] = []
    bidirectional_descriptor_qa_target_ranks: list[int] = []
    bidirectional_descriptor_qa_target_survivals: list[int] = []
    lexicon_descriptor_candidate_counts: list[int] = []
    lexicon_descriptor_rerank_counts: list[int] = []
    lexicon_descriptor_target_ranks: list[int] = []
    lexicon_descriptor_target_survivals: list[int] = []
    induced_map_descriptor_candidate_counts: list[int] = []
    induced_map_descriptor_rerank_counts: list[int] = []
    induced_map_descriptor_target_ranks: list[int] = []
    induced_map_descriptor_target_survivals: list[int] = []
    external_support_induced_map_descriptor_candidate_counts: list[int] = []
    external_support_induced_map_descriptor_rerank_counts: list[int] = []
    external_support_induced_map_descriptor_target_ranks: list[int] = []
    external_support_induced_map_descriptor_target_survivals: list[int] = []
    strict_induced_map_descriptor_candidate_counts: list[int] = []
    strict_induced_map_descriptor_rerank_counts: list[int] = []
    strict_induced_map_descriptor_target_ranks: list[int] = []
    strict_induced_map_descriptor_target_survivals: list[int] = []
    fewshot_descriptor_candidate_counts: list[int] = []
    fewshot_descriptor_rerank_counts: list[int] = []
    fewshot_descriptor_target_ranks: list[int] = []
    fewshot_descriptor_target_survivals: list[int] = []
    forced_choice_descriptor_candidate_counts: list[int] = []
    forced_choice_descriptor_rerank_counts: list[int] = []
    forced_choice_descriptor_target_ranks: list[int] = []
    forced_choice_descriptor_target_survivals: list[int] = []
    pairwise_tournament_descriptor_candidate_counts: list[int] = []
    pairwise_tournament_descriptor_rerank_counts: list[int] = []
    pairwise_tournament_descriptor_target_ranks: list[int] = []
    pairwise_tournament_descriptor_target_survivals: list[int] = []
    source_copy_guarded_descriptor_candidate_counts: list[int] = []
    source_copy_guarded_descriptor_rerank_counts: list[int] = []
    source_copy_guarded_descriptor_target_ranks: list[int] = []
    source_copy_guarded_descriptor_target_survivals: list[int] = []
    contrastive_descriptor_candidate_counts: list[int] = []
    contrastive_descriptor_rerank_counts: list[int] = []
    contrastive_descriptor_target_ranks: list[int] = []
    contrastive_descriptor_target_survivals: list[int] = []
    diagnostic_summaries: list[dict[str, float | int]] = []
    contrastive_diagnostic_summaries: list[dict[str, float | int]] = []
    fewshot_diagnostic_summaries: list[dict[str, float | int]] = []
    forced_choice_diagnostic_summaries: list[dict[str, float | int]] = []
    pairwise_tournament_diagnostic_summaries: list[dict[str, float | int]] = []
    induced_map_diagnostic_summaries: list[dict[str, float | int]] = []
    external_support_induced_map_diagnostic_summaries: list[dict[str, float | int]] = []
    full_lengths: list[int] = []
    target_position_fracs = [
        item.target_position_index / max(1, item.record_count - 1) for item in items
    ]
    query_position_fracs = [
        item.query_position_index / max(1, item.record_count - 1) for item in items
    ]
    target_query_distance_fracs = [
        abs(item.target_position_index - item.query_position_index)
        / max(1, item.record_count - 1)
        for item in items
    ]

    for item_idx, item in enumerate(items, start=1):
        print(f"evaluating item {item_idx}/{len(items)}", file=sys.stderr, flush=True)
        full_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        full_lengths.append(len(full_ids))
        if getattr(args, "diagnose_final_scores", False):
            diagnostic_summary, diagnostic_trace = bidirectional_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            diagnostic_summaries.append(diagnostic_summary)
            print(
                "diagnostic_final_scores "
                + json.dumps({"item": item_idx, **diagnostic_trace}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_contrastive_scores", False):
            (
                contrastive_diagnostic_summary,
                contrastive_diagnostic_trace,
            ) = contrastive_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            contrastive_diagnostic_summaries.append(contrastive_diagnostic_summary)
            print(
                "diagnostic_contrastive_scores "
                + json.dumps(
                    {"item": item_idx, **contrastive_diagnostic_trace},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_fewshot_scores", False):
            (
                fewshot_diagnostic_summary,
                fewshot_diagnostic_trace,
            ) = fewshot_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            fewshot_diagnostic_summaries.append(fewshot_diagnostic_summary)
            print(
                "diagnostic_fewshot_scores "
                + json.dumps(
                    {"item": item_idx, **fewshot_diagnostic_trace},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_forced_choice_scores", False):
            (
                forced_choice_diagnostic_summary,
                forced_choice_diagnostic_trace,
            ) = forced_choice_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            forced_choice_diagnostic_summaries.append(forced_choice_diagnostic_summary)
            print(
                "diagnostic_forced_choice_scores "
                + json.dumps(
                    {"item": item_idx, **forced_choice_diagnostic_trace},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_pairwise_tournament_scores", False):
            (
                pairwise_tournament_diagnostic_summary,
                pairwise_tournament_diagnostic_trace,
            ) = pairwise_tournament_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            pairwise_tournament_diagnostic_summaries.append(
                pairwise_tournament_diagnostic_summary
            )
            print(
                "diagnostic_pairwise_tournament_scores "
                + json.dumps(
                    {"item": item_idx, **pairwise_tournament_diagnostic_trace},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_induced_map_scores", False):
            (
                induced_map_diagnostic_summary,
                induced_map_diagnostic_trace,
            ) = induced_map_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
            )
            induced_map_diagnostic_summaries.append(induced_map_diagnostic_summary)
            print(
                "diagnostic_induced_map_scores "
                + json.dumps(
                    {"item": item_idx, **induced_map_diagnostic_trace},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "diagnose_external_support_induced_map_scores", False):
            (
                external_support_induced_map_diagnostic_summary,
                external_support_induced_map_diagnostic_trace,
            ) = induced_map_final_score_diagnostic(
                tokenizer,
                model,
                device,
                item,
                args.prefilter_candidates,
                getattr(args, "diagnostic_top_k", 5),
                map_source="external_support",
            )
            external_support_induced_map_diagnostic_summaries.append(
                external_support_induced_map_diagnostic_summary
            )
            print(
                "diagnostic_external_support_induced_map_scores "
                + json.dumps(
                    {
                        "item": item_idx,
                        **external_support_induced_map_diagnostic_trace,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        for policy in policies:
            print(
                f"  policy {policy} item={item_idx}/{len(items)} start",
                file=sys.stderr,
                flush=True,
            )
            if policy == "lm_bridge":
                kept_ids = lm_bridge_token_ids(tokenizer, model, device, item)
            elif policy == "phrase_lm_bridge":
                kept_ids = phrase_lm_bridge_token_ids(tokenizer, model, device, item)
            elif policy == "prefilter_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = prefilter_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                candidate_counts.append(candidate_count)
                rerank_counts.append(rerank_count)
                target_ranks.append(target_rank)
                target_survivals.append(target_survived)
            elif policy == "prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                compressed_candidate_counts.append(candidate_count)
                compressed_rerank_counts.append(rerank_count)
                compressed_target_ranks.append(target_rank)
                compressed_target_survivals.append(target_survived)
            elif policy == "margin_guarded_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                    override,
                ) = margin_guarded_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                guarded_candidate_counts.append(candidate_count)
                guarded_rerank_counts.append(rerank_count)
                guarded_target_ranks.append(target_rank)
                guarded_target_survivals.append(target_survived)
                guarded_overrides.append(override)
            elif policy == "reverse_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = reverse_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                reverse_descriptor_candidate_counts.append(candidate_count)
                reverse_descriptor_rerank_counts.append(rerank_count)
                reverse_descriptor_target_ranks.append(target_rank)
                reverse_descriptor_target_survivals.append(target_survived)
            elif policy == "bidirectional_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = bidirectional_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                bidirectional_descriptor_candidate_counts.append(candidate_count)
                bidirectional_descriptor_rerank_counts.append(rerank_count)
                bidirectional_descriptor_target_ranks.append(target_rank)
                bidirectional_descriptor_target_survivals.append(target_survived)
            elif policy == "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                bidirectional_descriptor_qa_candidate_counts.append(candidate_count)
                bidirectional_descriptor_qa_rerank_counts.append(rerank_count)
                bidirectional_descriptor_qa_target_ranks.append(target_rank)
                bidirectional_descriptor_qa_target_survivals.append(target_survived)
            elif policy == "lexicon_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = lexicon_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                lexicon_descriptor_candidate_counts.append(candidate_count)
                lexicon_descriptor_rerank_counts.append(rerank_count)
                lexicon_descriptor_target_ranks.append(target_rank)
                lexicon_descriptor_target_survivals.append(target_survived)
            elif policy == "induced_map_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                induced_map_descriptor_candidate_counts.append(candidate_count)
                induced_map_descriptor_rerank_counts.append(rerank_count)
                induced_map_descriptor_target_ranks.append(target_rank)
                induced_map_descriptor_target_survivals.append(target_survived)
            elif policy == "external_support_induced_map_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                external_support_induced_map_descriptor_candidate_counts.append(
                    candidate_count
                )
                external_support_induced_map_descriptor_rerank_counts.append(
                    rerank_count
                )
                external_support_induced_map_descriptor_target_ranks.append(target_rank)
                external_support_induced_map_descriptor_target_survivals.append(
                    target_survived
                )
            elif policy == "strict_induced_map_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = strict_induced_map_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                strict_induced_map_descriptor_candidate_counts.append(candidate_count)
                strict_induced_map_descriptor_rerank_counts.append(rerank_count)
                strict_induced_map_descriptor_target_ranks.append(target_rank)
                strict_induced_map_descriptor_target_survivals.append(target_survived)
            elif policy == "fewshot_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = fewshot_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                fewshot_descriptor_candidate_counts.append(candidate_count)
                fewshot_descriptor_rerank_counts.append(rerank_count)
                fewshot_descriptor_target_ranks.append(target_rank)
                fewshot_descriptor_target_survivals.append(target_survived)
            elif policy == "forced_choice_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = forced_choice_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                forced_choice_descriptor_candidate_counts.append(candidate_count)
                forced_choice_descriptor_rerank_counts.append(rerank_count)
                forced_choice_descriptor_target_ranks.append(target_rank)
                forced_choice_descriptor_target_survivals.append(target_survived)
            elif policy == "pairwise_tournament_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                pairwise_tournament_descriptor_candidate_counts.append(candidate_count)
                pairwise_tournament_descriptor_rerank_counts.append(rerank_count)
                pairwise_tournament_descriptor_target_ranks.append(target_rank)
                pairwise_tournament_descriptor_target_survivals.append(target_survived)
            elif policy == "source_copy_guarded_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                source_copy_guarded_descriptor_candidate_counts.append(candidate_count)
                source_copy_guarded_descriptor_rerank_counts.append(rerank_count)
                source_copy_guarded_descriptor_target_ranks.append(target_rank)
                source_copy_guarded_descriptor_target_survivals.append(target_survived)
            elif policy == "contrastive_descriptor_prefilter_compressed_lm_bridge":
                (
                    kept_ids,
                    candidate_count,
                    rerank_count,
                    target_rank,
                    target_survived,
                ) = contrastive_descriptor_prefilter_compressed_lm_bridge_token_ids(
                    tokenizer,
                    model,
                    device,
                    item,
                    args.prefilter_candidates,
                )
                contrastive_descriptor_candidate_counts.append(candidate_count)
                contrastive_descriptor_rerank_counts.append(rerank_count)
                contrastive_descriptor_target_ranks.append(target_rank)
                contrastive_descriptor_target_survivals.append(target_survived)
            elif policy == "compressed_lm_bridge":
                kept_ids = compressed_lm_bridge_token_ids(tokenizer, model, device, item)
            elif policy == "embedding_bridge":
                kept_ids = embedding_bridge_token_ids(tokenizer, model, device, item)
            else:
                kept_ids = policy_token_ids(
                    tokenizer,
                    item,
                    full_ids,
                    policy,
                    args.truncate_tokens,
                    args.sink_tokens,
                    args.recent_tokens,
                    args.lexical_records,
                )
            kept_fracs[policy].append(len(kept_ids) / max(1, len(full_ids)))
            if args.scoring == "rank":
                prediction = rank_answer(
                    tokenizer,
                    model,
                    device,
                    kept_ids,
                    item.candidates,
                )
            else:
                prediction = first_int(
                    answer_one(tokenizer, model, device, kept_ids, args.max_new_tokens)
                )
            correct[policy] += int(prediction == item.answer)
            progress_parts = [
                f"  policy {policy} item={item_idx}/{len(items)} done",
                f"kept={len(kept_ids)}/{len(full_ids)}",
                f"prediction={prediction}",
                f"answer={item.answer}",
            ]
            if policy in {
                "prefilter_lm_bridge",
                "prefilter_compressed_lm_bridge",
                "margin_guarded_prefilter_compressed_lm_bridge",
                "reverse_descriptor_prefilter_compressed_lm_bridge",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge",
                "lexicon_descriptor_prefilter_compressed_lm_bridge",
                "induced_map_descriptor_prefilter_compressed_lm_bridge",
                "external_support_induced_map_descriptor_prefilter_compressed_lm_bridge",
                "strict_induced_map_descriptor_prefilter_compressed_lm_bridge",
                "fewshot_descriptor_prefilter_compressed_lm_bridge",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge",
                "pairwise_tournament_descriptor_prefilter_compressed_lm_bridge",
                "source_copy_guarded_descriptor_prefilter_compressed_lm_bridge",
                "contrastive_descriptor_prefilter_compressed_lm_bridge",
            }:
                progress_parts.extend(
                    [
                        f"candidates={candidate_count}",
                        f"rerank={rerank_count}",
                        f"target_rank={target_rank}",
                        f"target_survived={target_survived}",
                    ]
                )
            print(" ".join(progress_parts), file=sys.stderr, flush=True)

    metrics: dict[str, float | int | str] = {
        "eval_completed": 1,
        "seed": args.seed,
        "items": args.items,
        "distractors": args.distractors,
        "model": args.model,
        "scoring": args.scoring,
        "task": args.task,
        "target_position": args.target_position,
        "lexical_records": args.lexical_records,
        "prefilter_candidates": args.prefilter_candidates,
        "mean_target_position_fraction": float(np.mean(target_position_fracs)),
        "mean_query_position_fraction": float(np.mean(query_position_fracs)),
        "mean_target_query_distance_fraction": float(
            np.mean(target_query_distance_fracs)
        ),
        "mean_full_tokens": float(np.mean(full_lengths)),
    }
    for policy in policies:
        metrics[f"{policy}_accuracy"] = correct[policy] / args.items
        metrics[f"{policy}_kept_fraction"] = float(np.mean(kept_fracs[policy]))
    for policy in policies:
        if policy == "full":
            continue
        metrics[f"{policy}_gap_vs_full"] = (
            metrics["full_accuracy"] - metrics[f"{policy}_accuracy"]
        )
    if candidate_counts:
        metrics["prefilter_lm_bridge_candidate_count"] = float(np.mean(candidate_counts))
        metrics["prefilter_lm_bridge_rerank_count"] = float(np.mean(rerank_counts))
        metrics["prefilter_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for rerank_count, candidate_count in zip(rerank_counts, candidate_counts)
                ]
            )
        )
        metrics["prefilter_lm_bridge_mean_target_rank"] = float(np.mean(target_ranks))
        metrics["prefilter_lm_bridge_min_target_rank"] = float(np.min(target_ranks))
        metrics["prefilter_lm_bridge_max_target_rank"] = float(np.max(target_ranks))
        metrics["prefilter_lm_bridge_target_survival_rate"] = float(
            np.mean(target_survivals)
        )
    if compressed_candidate_counts:
        metrics["prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(compressed_candidate_counts)
        )
        metrics["prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(compressed_rerank_counts)
        )
        metrics["prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for rerank_count, candidate_count in zip(
                        compressed_rerank_counts, compressed_candidate_counts
                    )
                ]
            )
        )
        metrics["prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(compressed_target_ranks)
        )
        metrics["prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(compressed_target_ranks)
        )
        metrics["prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(compressed_target_ranks)
        )
        metrics["prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(compressed_target_survivals)
        )
    if guarded_candidate_counts:
        metrics["margin_guarded_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(guarded_candidate_counts)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(guarded_rerank_counts)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for rerank_count, candidate_count in zip(
                        guarded_rerank_counts, guarded_candidate_counts
                    )
                ]
            )
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(guarded_target_ranks)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(guarded_target_ranks)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(guarded_target_ranks)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(guarded_target_survivals)
        )
        metrics["margin_guarded_prefilter_compressed_lm_bridge_override_rate"] = float(
            np.mean(guarded_overrides)
        )
    if reverse_descriptor_candidate_counts:
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(reverse_descriptor_candidate_counts)
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(reverse_descriptor_rerank_counts)
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        reverse_descriptor_candidate_counts,
                        reverse_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(reverse_descriptor_target_ranks)
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(reverse_descriptor_target_ranks)
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(reverse_descriptor_target_ranks)
        )
        metrics["reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(reverse_descriptor_target_survivals)
        )
    if bidirectional_descriptor_candidate_counts:
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(bidirectional_descriptor_candidate_counts)
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(bidirectional_descriptor_rerank_counts)
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        bidirectional_descriptor_candidate_counts,
                        bidirectional_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(bidirectional_descriptor_target_ranks)
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(bidirectional_descriptor_target_ranks)
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(bidirectional_descriptor_target_ranks)
        )
        metrics["bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(bidirectional_descriptor_target_survivals)
        )
    if bidirectional_descriptor_qa_candidate_counts:
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(bidirectional_descriptor_qa_candidate_counts)
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(bidirectional_descriptor_qa_rerank_counts)
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        bidirectional_descriptor_qa_candidate_counts,
                        bidirectional_descriptor_qa_rerank_counts,
                    )
                ]
            )
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(bidirectional_descriptor_qa_target_ranks)
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(bidirectional_descriptor_qa_target_ranks)
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(bidirectional_descriptor_qa_target_ranks)
        )
        metrics["bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(bidirectional_descriptor_qa_target_survivals)
        )
    if lexicon_descriptor_candidate_counts:
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(lexicon_descriptor_candidate_counts)
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(lexicon_descriptor_rerank_counts)
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        lexicon_descriptor_candidate_counts,
                        lexicon_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(lexicon_descriptor_target_ranks)
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(lexicon_descriptor_target_ranks)
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(lexicon_descriptor_target_ranks)
        )
        metrics["lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(lexicon_descriptor_target_survivals)
        )
    if induced_map_descriptor_candidate_counts:
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(induced_map_descriptor_candidate_counts)
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(induced_map_descriptor_rerank_counts)
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for candidate_count, rerank_count in zip(
                        induced_map_descriptor_candidate_counts,
                        induced_map_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(induced_map_descriptor_target_ranks)
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(induced_map_descriptor_target_ranks)
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(induced_map_descriptor_target_ranks)
        )
        metrics["induced_map_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(induced_map_descriptor_target_survivals)
        )
    if external_support_induced_map_descriptor_candidate_counts:
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(external_support_induced_map_descriptor_candidate_counts)
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(external_support_induced_map_descriptor_rerank_counts)
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for candidate_count, rerank_count in zip(
                        external_support_induced_map_descriptor_candidate_counts,
                        external_support_induced_map_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(external_support_induced_map_descriptor_target_ranks)
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(external_support_induced_map_descriptor_target_ranks)
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(external_support_induced_map_descriptor_target_ranks)
        )
        metrics["external_support_induced_map_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(external_support_induced_map_descriptor_target_survivals)
        )
    if strict_induced_map_descriptor_candidate_counts:
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(strict_induced_map_descriptor_candidate_counts)
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(strict_induced_map_descriptor_rerank_counts)
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank_count / max(1, candidate_count)
                    for candidate_count, rerank_count in zip(
                        strict_induced_map_descriptor_candidate_counts,
                        strict_induced_map_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(strict_induced_map_descriptor_target_ranks)
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(strict_induced_map_descriptor_target_ranks)
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(strict_induced_map_descriptor_target_ranks)
        )
        metrics["strict_induced_map_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(strict_induced_map_descriptor_target_survivals)
        )
    if fewshot_descriptor_candidate_counts:
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(fewshot_descriptor_candidate_counts)
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(fewshot_descriptor_rerank_counts)
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        fewshot_descriptor_candidate_counts,
                        fewshot_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(fewshot_descriptor_target_ranks)
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(fewshot_descriptor_target_ranks)
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(fewshot_descriptor_target_ranks)
        )
        metrics["fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(fewshot_descriptor_target_survivals)
        )
    if forced_choice_descriptor_candidate_counts:
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(forced_choice_descriptor_candidate_counts)
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(forced_choice_descriptor_rerank_counts)
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        forced_choice_descriptor_candidate_counts,
                        forced_choice_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(forced_choice_descriptor_target_ranks)
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(forced_choice_descriptor_target_ranks)
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(forced_choice_descriptor_target_ranks)
        )
        metrics["forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(forced_choice_descriptor_target_survivals)
        )
    if pairwise_tournament_descriptor_candidate_counts:
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(pairwise_tournament_descriptor_candidate_counts)
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(pairwise_tournament_descriptor_rerank_counts)
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        pairwise_tournament_descriptor_candidate_counts,
                        pairwise_tournament_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(pairwise_tournament_descriptor_target_ranks)
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(pairwise_tournament_descriptor_target_ranks)
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(pairwise_tournament_descriptor_target_ranks)
        )
        metrics["pairwise_tournament_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(pairwise_tournament_descriptor_target_survivals)
        )
    if source_copy_guarded_descriptor_candidate_counts:
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(source_copy_guarded_descriptor_candidate_counts)
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(source_copy_guarded_descriptor_rerank_counts)
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        source_copy_guarded_descriptor_candidate_counts,
                        source_copy_guarded_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(source_copy_guarded_descriptor_target_ranks)
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(source_copy_guarded_descriptor_target_ranks)
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(source_copy_guarded_descriptor_target_ranks)
        )
        metrics["source_copy_guarded_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(source_copy_guarded_descriptor_target_survivals)
        )
    if contrastive_descriptor_candidate_counts:
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_candidate_count"] = float(
            np.mean(contrastive_descriptor_candidate_counts)
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_count"] = float(
            np.mean(contrastive_descriptor_rerank_counts)
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
            np.mean(
                [
                    rerank / max(1, candidate)
                    for candidate, rerank in zip(
                        contrastive_descriptor_candidate_counts,
                        contrastive_descriptor_rerank_counts,
                    )
                ]
            )
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"] = float(
            np.mean(contrastive_descriptor_target_ranks)
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_min_target_rank"] = float(
            np.min(contrastive_descriptor_target_ranks)
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_max_target_rank"] = float(
            np.max(contrastive_descriptor_target_ranks)
        )
        metrics["contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
            np.mean(contrastive_descriptor_target_survivals)
        )
    if diagnostic_summaries:
        for key in sorted(diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean([float(summary[key]) for summary in diagnostic_summaries])
            )
    if contrastive_diagnostic_summaries:
        for key in sorted(contrastive_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in contrastive_diagnostic_summaries
                    ]
                )
            )
    if fewshot_diagnostic_summaries:
        for key in sorted(fewshot_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in fewshot_diagnostic_summaries
                    ]
                )
            )
    if forced_choice_diagnostic_summaries:
        for key in sorted(forced_choice_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in forced_choice_diagnostic_summaries
                    ]
                )
            )
    if pairwise_tournament_diagnostic_summaries:
        for key in sorted(pairwise_tournament_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in pairwise_tournament_diagnostic_summaries
                    ]
                )
            )
    if induced_map_diagnostic_summaries:
        for key in sorted(induced_map_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in induced_map_diagnostic_summaries
                    ]
                )
            )
    if external_support_induced_map_diagnostic_summaries:
        for key in sorted(external_support_induced_map_diagnostic_summaries[0]):
            metrics[f"mean_{key}"] = float(
                np.mean(
                    [
                        float(summary[key])
                        for summary in external_support_induced_map_diagnostic_summaries
                    ]
                )
            )
    return metrics


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    tokenizer, model, device = load_model(args)
    return evaluate_with_model(args, tokenizer, model, device)


def write_metrics(metrics: dict[str, float | int | str], metrics_path: str | None) -> None:
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if not metrics_path:
        return
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(rendered + "\n")


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    write_metrics(metrics, args.metrics_path)


if __name__ == "__main__":
    main()
