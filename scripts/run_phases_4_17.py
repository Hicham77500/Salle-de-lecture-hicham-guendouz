#!/usr/bin/env python3
"""Exécute les phases 4-17 et met à jour RAPPORT.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def update_rapport(results: dict) -> None:
    p4 = results.get("phase4", {})
    p5 = results.get("phase5", {})
    p67 = results.get("phase6_7", {})
    p8 = results.get("phase8", {})
    p9 = results.get("phase9", {})
    p1013 = results.get("phase10_13", {})
    p1417 = results.get("phase14_17", {})

    rapport_path = ROOT / "RAPPORT.md"
    text = rapport_path.read_text(encoding="utf-8")

    def replace_section(marker: str, content: str) -> None:
        nonlocal text
        start = text.find(marker)
        if start == -1:
            return
        end = text.find("\n---", start + 1)
        if end == -1:
            end = text.find("\n### Phase", start + 10)
        if end == -1:
            end = text.find("\n## Acte", start + 10)
        if end == -1:
            end = len(text)
        text = text[:start] + marker + "\n\n" + content + "\n" + text[end:]

    if p4:
        pannes = p4.get("pannes", [])
        lines = ["| # | Panne | Geste | Test 1 min |", "|---|-------|-------|------------|"]
        for i, p in enumerate(pannes, 1):
            nom = p.get("nom") or p.get("name", "")
            geste = p.get("geste", "")
            test = p.get("test_1min") or p.get("test_rapide", "")
            lines.append(f"| {i} | {nom} | {geste} | {test} |")
        lines.append("\n**Figure** : `reports/figures/phase4_pannes.png`")
        replace_section("### Phase 4 — Carnet de pannes", "\n".join(lines))

    if p5:
        content = f"""| Version | Temps (s) | Val acc |
|---------|-----------|---------|
| Baseline | {p5['baseline']['total_seconds']:.1f} | {p5['baseline']['final_val_acc']:.3f} |
| Optimisé | {p5['optimized']['total_seconds']:.1f} | {p5['optimized']['final_val_acc']:.3f} |

**Facteur d'accélération** : ×{p5['speedup_factor']:.1f}

**Figure** : `reports/figures/phase5_benchmark.png`
"""
        replace_section("### Phase 5 — Budget de calcul", content)

    if p67:
        p6, p7 = p67.get("phase6", {}), p67.get("phase7", {})
        rf = p6.get("receptive_field_table", [])
        tbl = "| Couche | Étendue | Cumul |\n|--------|---------|-------|\n"
        for row in rf:
            tbl += f"| {row['couche']} | {row['etendue']} | {row['cumul']} |\n"
        content = f"""**Longueur max (jetons)** : {p6.get('max_tokens')} | **Médiane** : {p6.get('median_tokens')}

{tbl}

**Total RF ≥ max** : {p6.get('covers_max')} | **Perturbation 1er mot** : Δ={p6.get('perturbation_delta', 0):.4f}

**Phase 7** : {p7.get('fix')} — acc batch=4 : {p7.get('batch4_after_acc', 0):.3f}

**Figure** : `reports/figures/phase7_batch4.png`
"""
        replace_section("### Phase 6 — Champ de vision", content)

    if p8:
        content = f"""**Mots interdits** : {p8['n_forbidden_words']} | **Restants après masque** : {p8['remaining_with_forbidden']} (attendu 0)

| Métrique | Avant | Après |
|----------|-------|-------|
| Accuracy sklearn | {p8['before']['sklearn']['accuracy']:.3f} | {p8['after']['sklearn']['accuracy']:.3f} |
| Macro-F1 PyTorch | {p8['before']['pytorch']['macro_f1']:.3f} | {p8['after']['pytorch']['macro_f1']:.3f} |
"""
        replace_section("### Phase 8 — Masque vocabulaire formes", content)

    if p9:
        lines = []
        for c in p9.get("cases", []):
            top = ", ".join(f"{a['mot']}({a['poids']:.2f})" for a in c["attribution"][:5])
            lines.append(f"#### {c['type'].title()} — prédit `{c['forme_predite']}` / vrai `{c['vraie_forme']}`\n\n*{c['temoignage'][:120]}...*\n\nMots clés : {top}\n")
        replace_section("### Phase 9 — Trois explications", "\n".join(lines))

    if p1013:
        p10, p11, p12, p13 = p1013.get("phase10", {}), p1013.get("phase11", {}), p1013.get("phase12", {}), p1013.get("phase13", {})
        content = f"""**Relevé** : « {p10.get('snippet', '')[:80]}... »

Tokens : {', '.join(p10.get('tokens', [])[:10])}

**Phase 11** — écart permuté avant pos : {p11.get('diff_before_pos', 0):.4f} → après : {p11.get('diff_after_pos', 0):.4f}

**Phase 12** — facteur doublement ~{p12.get('doubling_factor_128_64', 0):.1f}×

**Phase 13** — désaccord têtes : {p13.get('disagree', 0):.4f} (contrôle identique : {p13.get('disagree_identical_heads', 0):.4f})

**Figures** : phase10_attention.png, phase12_benchmark.png, phase13_multihead.png
"""
        replace_section("### Phase 10 — Single-head manuel", content)

    if p1417:
        p14 = p1417.get("phase14", {})
        p15 = p1417.get("phase15", {})
        p16 = p1417.get("phase16", {})
        p17 = p1417.get("phase17", {})
        frozen = p14.get("regimes", {}).get("frozen", {}) if isinstance(p14, dict) else {}
        content = f"""| Régime | Accuracy | Params entraînés | Temps |
|--------|----------|-------------------|-------|
| Frozen DistilBERT | {frozen.get('metrics', {}).get('accuracy', 'N/A')} | {frozen.get('params_trained', 'N/A')} | {frozen.get('train_seconds', 'N/A')} |

**Phase 15** — budget {p15.get('budget_tokens')} tokens | {len(p15.get('questions', []))} questions

**Phase 16** — marge {p16.get('margin_score_announced')} | disque {p16.get('disk_mb_before', 0):.1f}→{p16.get('disk_mb_after_estimated', 0):.1f} Mo

**Phase 17** — poids non modifiés : {p17.get('weights_modified')} | tri aveugle : {p17.get('blind_sort_accuracy', 0):.0%}
"""
        replace_section("### Phase 14 — Modèle emprunté", content)

    rapport_path.write_text(text, encoding="utf-8")


def main():
    results = {}
    print("=== Phase 4 ===")
    from src.shape.debug import run_phase4
    results["phase4"] = run_phase4()

    print("=== Phase 5 ===")
    from src.shape.benchmark import run_phase5
    results["phase5"] = run_phase5()

    print("=== Phases 6-7 ===")
    from src.shape.receptive_field import run_phases_6_7
    results["phase6_7"] = run_phases_6_7()

    print("=== Phase 8 ===")
    from src.shape.mask_vocab import run_phase8
    results["phase8"] = run_phase8(n_epochs=15)

    print("=== Phase 9 ===")
    from src.shape.explain import run_phase9
    results["phase9"] = run_phase9()

    print("=== Phases 10-13 ===")
    from src.attention.run_phases import run_attention_phases
    results["phase10_13"] = run_attention_phases()

    print("=== Phases 14-17 ===")
    from src.transfer.run_phases import run_phases_14_17
    results["phase14_17"] = run_phases_14_17()

    update_rapport(results)
    with open(ROOT / "reports" / "phases_4_17_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: "done" for k in results}, f)
    print("\nPhases 4-17 terminées. RAPPORT.md mis à jour.")


if __name__ == "__main__":
    main()
