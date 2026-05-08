#!/bin/bash
# Patch pour tester: pas de pénalité sur COARSE=NONE
# Applique --min-coarse-none-weight=0.0 au training

cd "$(dirname "$0")"

echo "========================================"
echo "📝 PATCH: Désactivation pénalité COARSE=NONE"
echo "========================================"
echo ""
echo "Modification: --min-coarse-none-weight 0.0"
echo "   → Les erreurs de prédiction sur NONE ne pénalisent plus le modèle"
echo "   → Devrait améliorer les F1 des vraies classes (PER/LOC/ORG/TIME/EVENT)"
echo ""

# Backup
if [ ! -f run_adaptive_training.sh.backup ]; then
    cp run_adaptive_training.sh run_adaptive_training.sh.backup
    echo "✅ Backup créé: run_adaptive_training.sh.backup"
fi

# Chercher la ligne avec --min-coarse-none-weight et la remplacer par 0.0
# Sinon, l'ajouter après --class-weight-power
if grep -q "min-coarse-none-weight" run_adaptive_training.sh; then
    sed -i.tmp 's/--min-coarse-none-weight [0-9.]*/--min-coarse-none-weight 0.0/' run_adaptive_training.sh
    echo "✅ Paramètre modifié: --min-coarse-none-weight 0.0"
else
    # Ajouter après --class-weight-power
    sed -i.tmp '/--class-weight-power/a\        --min-coarse-none-weight 0.0 \\' run_adaptive_training.sh
    echo "✅ Paramètre ajouté: --min-coarse-none-weight 0.0"
fi

rm -f run_adaptive_training.sh.tmp

echo ""
echo "✅ Patch appliqué!"
echo ""
echo "Pour lancer le training patchés:"
echo "   bash run_adaptive_training.sh"
echo ""
echo "Pour restaurer l'original:"
echo "   cp run_adaptive_training.sh.backup run_adaptive_training.sh"
echo ""

