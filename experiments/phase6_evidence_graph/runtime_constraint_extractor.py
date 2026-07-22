"""Deterministic runtime constraint extraction for Phase 6-B."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


RULESET_VERSION = "phase6b-runtime-constraint-rules-v0.4"

_DAILY_DOSE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*mg\s*/\s*\(?\s*kg"
    r"\s*(?:(?:[./·∙*]\s*)?d|/\s*day)\s*\)?",
    re.IGNORECASE,
)
_SIMPLE_DOSE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*mg\s*/\s*kg"
    r"(?!\s*(?:(?:[./·∙*]\s*)?d|/\s*day))",
    re.IGNORECASE,
)
_MONITORING_WINDOW_PATTERN = re.compile(
    r"(?P<start>\d{1,3})\s*(?:-|~|—|–|至)\s*(?P<end>\d{1,3})"
    r"\s*(?:小时|h\b)",
    re.IGNORECASE,
)
_SINGLE_MONITORING_WINDOW_PATTERN = re.compile(
    r"(?<![-~—–至\d])(?P<value>\d{1,3})\s*(?:小时|h\b)"
    r"(?!\s*[-~—–至])",
    re.IGNORECASE,
)
_PATIENT_WEIGHT_PATTERN = re.compile(
    r"(?:体重\s*)?(?P<value>\d+(?:\.\d+)?)\s*kg\b",
    re.IGNORECASE,
)
_COMBINATION_CUE_PATTERN = re.compile(
    r"(?:同时(?:使用|服用|口服|吃)|联合(?:使用|用药)|多种药物|"
    r"\bboth agents\b|\bsimultaneously\b)",
    re.IGNORECASE,
)
_COMBINATION_FAMILY_PATTERNS = (
    (
        "combination_antimicrobial",
        re.compile(
            r"(?:阿奇霉素|头孢|青霉素|抗菌药物?|抗生素|"
            r"大环内酯|β-内酰胺|beta-lactam|antibiotic|antimicrobial)",
            re.IGNORECASE,
        ),
    ),
    (
        "combination_antipyretic",
        re.compile(
            r"(?:布洛芬|对乙酰氨基酚|扑热息痛|ibuprofen|paracetamol|"
            r"acetaminophen)",
            re.IGNORECASE,
        ),
    ),
    (
        "combination_cough_medicine",
        re.compile(
            r"(?:止咳药|镇咳药|化痰药|祛痰药|antitussive|expectorant)",
            re.IGNORECASE,
        ),
    ),
    (
        "combination_steroid",
        re.compile(
            r"(?:激素|糖皮质激素|肾上腺皮质激素|corticosteroid)",
            re.IGNORECASE,
        ),
    ),
    (
        "combination_nebulization",
        re.compile(
            r"(?:雾化(?:吸入|治疗)?|nebulized|nebulisation|nebulization)",
            re.IGNORECASE,
        ),
    ),
)

_ALIAS_PATTERNS = (
    (
        "frequency",
        "qd",
        re.compile(r"(?:\bqd\b|每日\s*(?:一|1)\s*次|一天\s*(?:一|1)\s*次)", re.IGNORECASE),
    ),
    (
        "frequency",
        "bid",
        re.compile(r"(?:\bbid\b|每日\s*(?:两|二|2)\s*次|一天\s*(?:两|二|2)\s*次)", re.IGNORECASE),
    ),
    (
        "route",
        "iv_infusion",
        re.compile(r"(?:静脉滴注|静脉输注|静滴|静点)"),
    ),
    (
        "route",
        "iv_unspecified",
        re.compile(r"(?:静脉使用|静脉给药)"),
    ),
    (
        "monitoring_trigger",
        "nonresponse",
        re.compile(r"(?:症状|病情|治疗|疗效)?(?:无|未见)(?:明显)?改善|疗效不佳"),
    ),
    (
        "monitoring_trigger",
        "persistent_fever",
        re.compile(r"(?:体温(?:仍|持续)(?:高|升高)|持续发热)"),
    ),
    (
        "monitoring_trigger",
        "relapse",
        re.compile(r"一度改善(?:后|又)?(?:恶化|加重)"),
    ),
    (
        "monitoring_action",
        "reassess",
        re.compile(
            r"(?:再次|重新)(?:进行)?(?:临床或实验室)?评估|复评"
        ),
    ),
    (
        "monitoring_action",
        "assess_effect",
        re.compile(r"(?:评价(?:药物)?疗效|疗效评价)"),
    ),
    (
        "monitoring_target",
        "hepatic_renal_function",
        re.compile(
            r"(?:肝肾功能|肝功能(?:和|与|及|、)?肾功能|"
            r"肾功能(?:和|与|及|、)?肝功能)"
        ),
    ),
    (
        "monitoring_target",
        "body_temperature",
        re.compile(r"(?:体温(?:情况|变化|仍高|持续升高|持续高)|持续发热)"),
    ),
    (
        "contraindication_check",
        "allergy_history",
        re.compile(r"(?:过敏史|药物过敏|对[^，。；]{1,20}过敏)"),
    ),
    (
        "contraindication_action",
        "avoid",
        re.compile(r"(?:禁用于|禁用|禁止使用|不得使用|不应使用)"),
    ),
    (
        "contraindication_action",
        "caution",
        re.compile(r"(?:慎用|慎重|谨慎使用)"),
    ),
    (
        "age_group",
        "infant",
        re.compile(r"(?:婴儿|婴幼儿)"),
    ),
    (
        "population_context",
        "immunodeficiency",
        re.compile(r"(?:免疫缺陷|免疫功能低下)"),
    ),
    (
        "interaction_check",
        "drug_interaction",
        re.compile(r"(?:药物)?相互作用"),
    ),
    (
        "evidence_scope",
        "dose_guidance",
        re.compile(r"(?:剂量依据|推荐剂量|给药剂量|剂量为)"),
    ),
    (
        "evidence_scope",
        "formulation_listing",
        re.compile(
            r"(?:剂型[、，及和/ ]*规格|片剂、胶囊|品种名称.{0,20}剂型)"
        ),
    ),
    (
        "dose_adjustment",
        "increase",
        re.compile(r"(?:加大|增加|提高)(?:抗菌药物?|抗生素)?剂量"),
    ),
    (
        "dose_adjustment",
        "insufficient",
        re.compile(r"(?:抗菌药物?|抗生素)?剂量不足"),
    ),
    (
        "coverage_action",
        "broaden_pathogen_coverage",
        re.compile(
            r"(?:广\s*(?:泛\s*)?覆\s*盖|覆\s*盖)"
            r"\s*(?:潜在|可能)?\s*病原体"
        ),
    ),
)


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("runtime constraint input must be text")
    return unicodedata.normalize("NFKC", text).strip()


def _normalized_number(raw_value: str) -> str:
    value = float(raw_value)
    return str(int(value)) if value.is_integer() else format(value, "g")


def _add_constraint(
    found: dict[tuple[str, str], set[str]],
    *,
    constraint_type: str,
    normalized_value: str,
    surface_form: str,
) -> None:
    found.setdefault((constraint_type, normalized_value), set()).add(
        surface_form.strip()
    )


def extract_runtime_constraints(*texts: str | Iterable[str]) -> list[dict]:
    """Extract a stable, small constraint vocabulary from runtime text only."""
    flattened: list[str] = []
    for value in texts:
        if isinstance(value, str):
            flattened.append(value)
        elif isinstance(value, Iterable):
            flattened.extend(value)
        else:
            raise TypeError("runtime constraint input must be text or an iterable")

    found: dict[tuple[str, str], set[str]] = {}
    for raw_text in flattened:
        text = _normalize_text(raw_text)
        if not text:
            continue

        daily_spans: list[tuple[int, int]] = []
        dose_surface_forms: list[str] = []
        for match in _DAILY_DOSE_PATTERN.finditer(text):
            daily_spans.append(match.span())
            dose_surface_forms.append(match.group(0))
            _add_constraint(
                found,
                constraint_type="dose",
                normalized_value=f"{_normalized_number(match.group('value'))}mg/kg/day",
                surface_form=match.group(0),
            )

        for match in _SIMPLE_DOSE_PATTERN.finditer(text):
            if any(
                match.start() >= start and match.end() <= end
                for start, end in daily_spans
            ):
                continue
            dose_surface_forms.append(match.group(0))
            _add_constraint(
                found,
                constraint_type="dose",
                normalized_value=f"{_normalized_number(match.group('value'))}mg/kg",
                surface_form=match.group(0),
            )

        for match in _MONITORING_WINDOW_PATTERN.finditer(text):
            _add_constraint(
                found,
                constraint_type="monitoring_window",
                normalized_value=(
                    f"{int(match.group('start'))}-{int(match.group('end'))}h"
                ),
                surface_form=match.group(0),
            )

        for match in _SINGLE_MONITORING_WINDOW_PATTERN.finditer(text):
            _add_constraint(
                found,
                constraint_type="monitoring_window",
                normalized_value=f"{int(match.group('value'))}h",
                surface_form=match.group(0),
            )

        for match in _PATIENT_WEIGHT_PATTERN.finditer(text):
            _add_constraint(
                found,
                constraint_type="patient_weight",
                normalized_value=f"{_normalized_number(match.group('value'))}kg",
                surface_form=match.group(0),
            )

        for combination_cue in _COMBINATION_CUE_PATTERN.finditer(text):
            context_start = max(0, combination_cue.start() - 96)
            context_end = min(len(text), combination_cue.end() + 96)
            local_context = text[context_start:context_end]
            for constraint_type, family_pattern in _COMBINATION_FAMILY_PATTERNS:
                family_match = family_pattern.search(local_context)
                if family_match:
                    _add_constraint(
                        found,
                        constraint_type=constraint_type,
                        normalized_value="coadministration",
                        surface_form=(
                            f"{combination_cue.group(0)} / "
                            f"{family_match.group(0)}"
                        ),
                    )

        for surface_form in dose_surface_forms:
            _add_constraint(
                found,
                constraint_type="evidence_scope",
                normalized_value="dose_guidance",
                surface_form=surface_form,
            )

        for constraint_type, normalized_value, pattern in _ALIAS_PATTERNS:
            for match in pattern.finditer(text):
                _add_constraint(
                    found,
                    constraint_type=constraint_type,
                    normalized_value=normalized_value,
                    surface_form=match.group(0),
                )

    return [
        {
            "constraint_type": constraint_type,
            "normalized_value": normalized_value,
            "surface_forms": sorted(surface_forms),
            "ruleset_version": RULESET_VERSION,
        }
        for (constraint_type, normalized_value), surface_forms in sorted(
            found.items()
        )
    ]
