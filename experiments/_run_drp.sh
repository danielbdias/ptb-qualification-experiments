DOMAIN=$1

echo ""
echo "--------------------------------------------------------------------------------"
echo "Running experiments for domain: $DOMAIN"
echo "--------------------------------------------------------------------------------"
echo ""

PYTHONWARNINGS=ignore python ./experiments/$DOMAIN/no_heuristic_drp.py
PYTHONWARNINGS=ignore python ./experiments/$DOMAIN/heuristic_drp.py
# PYTHONWARNINGS=ignore python ./experiments/$DOMAIN/_graphs_for_drp.py