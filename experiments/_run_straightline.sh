DOMAIN=$1

echo ""
echo "--------------------------------------------------------------------------------"
echo "Running experiments for domain: $DOMAIN"
echo "--------------------------------------------------------------------------------"
echo ""

PYTHONWARNINGS=ignore python ./experiments/no_heuristic_straightline.py $DOMAIN
PYTHONWARNINGS=ignore python ./experiments/heuristic_straightline.py $DOMAIN
# PYTHONWARNINGS=ignore python ./experiments/$DOMAIN/_graphs_for_straightplan.py