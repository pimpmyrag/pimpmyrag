import wandb
import time
from datetime import datetime

api = wandb.Api()

print("=" * 100)
print("🔬 MATRICE DE TEST: PyTorch version vs Dataset version")
print("=" * 100)
print()
print("Hypothèse: Régression -7% causée par PyTorch 2.4 (vs 2.6 dans v8.0)")
print()

runs = [
    ("113w2omu", "v8.0 RÉFÉRENCE", "v8.0", "2.6+", "✅ SUCCÈS"),
    ("97kpgapu", "TEST 1", "v8.1", "2.4", "❌ RÉGRESSE -7%"),
    ("gqe8ouku7feqkb", "TEST 2", "v8.0", "2.4", "⏳ En cours"),
    ("mali2vprtnjtr8", "TEST 3", "v8.1", "2.6+", "⏳ En cours (CRITIQUE)"),
]

while True:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] État des runs:")
    print("=" * 100)
    print(f"{'Run':15s} {'Dataset':8s} {'PyTorch':8s} {'Epoch':>6s} {'Bound':>7s} {'Coarse':>7s} {'Fine':>7s} {'SVO':>7s} {'Status':15s}")
    print("-" * 100)

    for run_id, name, dataset, pytorch, status in runs:
        try:
            run = api.run(f"pimpmyrag-pimpmyrag/pimpmyrag-ner/{run_id}")
            summary = run.summary
            epoch = int(summary.get('epoch', 0))

            b = summary.get('test/boundary_f1', 0)
            c = summary.get('test/coarse_f1', 0)
            f = summary.get('test/fine_f1', 0)
            s = summary.get('test/svo_f1', 0)

            state = run.state
            if state == "finished":
                status_str = "✅ Fini"
            elif state == "running":
                status_str = f"⏳ Epoch {epoch}"
            else:
                status_str = f"❓ {state}"

            print(f"{name:15s} {dataset:8s} {pytorch:8s} {epoch:6d} {b:7.4f} {c:7.4f} {f:7.4f} {s:7.4f} {status_str:15s}")

        except Exception as e:
            print(f"{name:15s} {dataset:8s} {pytorch:8s}   N/A     -       -       -       -    ❌ Erreur")

    print()
    print("=" * 100)
    print("🎯 VERDICTS ATTENDUS:")
    print("=" * 100)

    try:
        # Check if TEST 3 (mali2vprtnjtr8) has reached epoch 7
        test3 = api.run(f"pimpmyrag-pimpmyrag/pimpmyrag-ner/mali2vprtnjtr8")
        test3_epoch = test3.summary.get('epoch', 0)

        if test3_epoch >= 7:
            history = test3.history(samples=2000)
            if 'epoch' in history.columns:
                e7 = history[history['epoch'] == 7.0]
                if len(e7) > 0:
                    row = e7.iloc[-1]
                    fine_7 = row.get('test/fine_f1', 0)

                    # Compare with v8.0 (113w2omu) at epoch 7
                    ref = api.run("pimpmyrag-pimpmyrag/pimpmyrag-ner/113w2omu")
                    ref_history = ref.history(samples=2000)
                    ref_e7 = ref_history[ref_history['epoch'] == 7.0]
                    ref_fine_7 = ref_e7.iloc[-1].get('test/fine_f1', 0) if len(ref_e7) > 0 else 0.607

                    delta = fine_7 - ref_fine_7

                    print(f"\n🔬 TEST 3 (v8.1 + torch 2.6) @ epoch 7:")
                    print(f"   Fine F1: {fine_7:.4f} vs {ref_fine_7:.4f} (v8.0) → {delta:+.4f}")
                    print()

                    if delta >= -0.02:  # Within 2% of v8.0
                        print("✅ VERDICT: PyTorch 2.4→2.6 EST LA CAUSE!")
                        print("   → Torch 2.6+ restaure les performances v8.0")
                        print("   → Dataset v8.1 (morpho) est OK")
                    else:
                        print("❌ VERDICT: Dataset v8.1 EST LA CAUSE!")
                        print("   → Même avec torch 2.6+, regression persiste")
                        print("   → Morpho/hint_rate ont un effet négatif")

        # Check TEST 2 (gqe8ouku7feqkb)
        test2 = api.run(f"pimpmyrag-pimpmyrag/pimpmyrag-ner/gqe8ouku7feqkb")
        test2_epoch = test2.summary.get('epoch', 0)

        if test2_epoch >= 7:
            history2 = test2.history(samples=2000)
            if 'epoch' in history2.columns:
                e7_2 = history2[history2['epoch'] == 7.0]
                if len(e7_2) > 0:
                    row2 = e7_2.iloc[-1]
                    fine_7_2 = row2.get('test/fine_f1', 0)

                    ref_fine_7 = 0.607
                    delta2 = fine_7_2 - ref_fine_7

                    print(f"\n🔬 TEST 2 (v8.0 + torch 2.4) @ epoch 7:")
                    print(f"   Fine F1: {fine_7_2:.4f} vs {ref_fine_7:.4f} (v8.0) → {delta2:+.4f}")

                    if delta2 >= -0.02:
                        print("   → v8.0 dataset fonctionne même avec torch 2.4")
                    else:
                        print("   → torch 2.4 dégrade même v8.0 dataset!")

    except:
        pass

    print()
    time.sleep(180)  # Check every 3 minutes

