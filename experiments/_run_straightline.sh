DOMAIN=$1

echo ""
echo "--------------------------------------------------------------------------------"
echo "Running experiments for domain: $DOMAIN"
echo "--------------------------------------------------------------------------------"
echo ""

cp ./experiments/$DOMAIN/_common_params.py ./experiments/_common_params.py

start=$(date +%s)

#PYTHONWARNINGS=ignore python ./experiments/no_heuristic_straightline.py $DOMAIN
PYTHONWARNINGS=ignore python ./experiments/heuristic_straightline.py $DOMAIN

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"