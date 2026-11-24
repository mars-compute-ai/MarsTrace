import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeliveredAnalysis:
    """The analysis a job delivers in ``marstrace.json`` beside its trace, read for a person.

    A fast job carries the analyser's summary of its first run: findings with where their sites
    sit, the causes the trace cannot settle with the measurement that would, and the balance of
    the grid. A job may also carry its agent's notes.

    Args:
        status: ``complete``, or ``skipped`` when the job delivered none
        reason: why it was skipped
        analyser: the analyser's release
        notes: the agent's notes, or None
        summary: the analyser's ``summary --json``
    """

    SECTION_TYPE = "marscompute.analysis"
    STATUS_COMPLETE = "complete"
    PERCENT = 100

    status: str
    reason: str = ""
    analyser: str | None = None
    notes: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_document(document: Any) -> "DeliveredAnalysis | None":
        """The analysis section of a decoded ``marstrace.json``, or None when it has none."""
        if not isinstance(document, dict) or not isinstance(document.get("sections"), list):
            return None
        for section in document["sections"]:
            if (
                isinstance(section, dict)
                and section.get("type") == DeliveredAnalysis.SECTION_TYPE
                and isinstance(section.get("data"), dict)
            ):
                data = section["data"]
                return DeliveredAnalysis(
                    status=str(data.get("status") or ""),
                    reason=str(data.get("reason") or ""),
                    analyser=data.get("analyser_release") if isinstance(data.get("analyser_release"), str) else None,
                    notes=data.get("notes") if isinstance(data.get("notes"), str) and data["notes"].strip() else None,
                    summary=data.get("summary") if isinstance(data.get("summary"), dict) else {},
                )
        return None

    @staticmethod
    def from_file(path: str) -> "DeliveredAnalysis | None":
        """The analysis of a ``marstrace.json`` on disk, or None when it carries none or does not read as JSON."""
        try:
            with open(
                file=path,
                mode="r",
                encoding="utf-8",
            ) as trace_file:
                document = json.load(trace_file)
        except (
            OSError,
            ValueError,
        ):
            return None
        return DeliveredAnalysis.from_document(document)

    @staticmethod
    def _share(value: Any) -> str | None:
        return f"{round(value * DeliveredAnalysis.PERCENT)}%" if isinstance(value, (int, float)) else None

    @staticmethod
    def location_text(location: dict[str, Any]) -> str:
        """A site as a reader names it: where the region crosses it, its instruction and source line."""
        list_part = []
        if isinstance(location.get("sass_index"), int):
            list_part.append(f"SASS {location['sass_index']}")
        if location.get("opcode"):
            list_part.append(str(location["opcode"]))
        if location.get("source"):
            list_part.append(f"({location['source']})")
        site = f"site {location['site_id']}" if isinstance(location.get("site_id"), int) else "site"
        detail = f": {' '.join(list_part)}" if list_part else ""
        return f"{location.get('boundary') or 'at'} {site}{detail}"

    def facts(self) -> list[tuple[str, str]]:
        """The grid's balance and the run's timing, as label and value."""
        imbalance = self.summary.get("imbalance") if isinstance(self.summary.get("imbalance"), dict) else {}
        spec = self.summary.get("spec") if isinstance(self.summary.get("spec"), dict) else {}
        gate = self.summary.get("gate") if isinstance(self.summary.get("gate"), dict) else {}
        list_fact = [
            ("SM utilization", DeliveredAnalysis._share(imbalance.get("sm_utilization"))),
            ("tail", DeliveredAnalysis._share(imbalance.get("tail_share"))),
            (
                "blocks resident per SM",
                f"{imbalance['resident_mean']:.2f}" if isinstance(imbalance.get("resident_mean"), (int, float)) else None,
            ),
            ("SMs", str(imbalance["sm_count"]) if isinstance(imbalance.get("sm_count"), int) else None),
            (
                "traced span",
                f"{self.summary['span']:,.0f} ns" if isinstance(self.summary.get("span"), (int, float)) else None,
            ),
            (
                "kernel time, uninstrumented",
                f"{spec['uninstrumented_us']:.1f} us" if isinstance(spec.get("uninstrumented_us"), (int, float)) else None,
            ),
            ("trace check", gate.get("verdict") if isinstance(gate.get("verdict"), str) else None),
        ]
        return [(label, value) for label, value in list_fact if value is not None]

    def text(self) -> str:
        """The analysis as text: facts, findings with where they sit, what the trace cannot tell, notes."""
        if self.status != DeliveredAnalysis.STATUS_COMPLETE:
            return f"analysis: none delivered ({self.reason or 'not completed'})\n"
        list_line = [f"analysis{f' (analyser {self.analyser})' if self.analyser else ''}"]
        for label, value in self.facts():
            list_line.append(f"  {label}: {value}")
        list_finding = [finding for finding in self.summary.get("findings") or [] if isinstance(finding, dict)]
        list_line.append("findings:" if list_finding else "findings: none flagged")
        for index, finding in enumerate(list_finding, start=1):
            list_line.append(f"  {index}. [{finding.get('code', '')}] {finding.get('title', '')}")
            list_location = [location for location in finding.get("list_location") or [] if isinstance(location, dict)]
            where = ", ".join(
                part
                for part in [
                    str(finding.get("region") or ""),
                    "; ".join(DeliveredAnalysis.location_text(location) for location in list_location),
                ]
                if part
            )
            if where:
                list_line.append(f"     where: {where}")
            if finding.get("evidence"):
                list_line.append(f"     evidence: {finding['evidence']}")
            if finding.get("next_step"):
                list_line.append(f"     next: {finding['next_step']}")
            for hypothesis in finding.get("list_hypothesis") or []:
                if isinstance(hypothesis, dict):
                    list_line.append(f"     - {hypothesis.get('cause', '')}: {hypothesis.get('verify_with', '')}")
        list_unmeasured = [entry for entry in self.summary.get("unmeasured") or [] if isinstance(entry, dict)]
        if list_unmeasured:
            list_line.append("what this trace cannot tell:")
            for entry in list_unmeasured:
                list_line.append(f"  - {entry.get('cause', '')}: {entry.get('verify_with', '')}")
        if self.notes:
            list_line.append("notes:")
            list_line.extend(f"  {line}" for line in self.notes.strip().splitlines())
        return "\n".join(list_line) + "\n"

    def to_json(self) -> dict[str, Any]:
        """The section's data as delivered, for a machine reader."""
        return {
            "status": self.status,
            "reason": self.reason or None,
            "analyser_release": self.analyser,
            "notes": self.notes,
            "summary": self.summary,
        }
