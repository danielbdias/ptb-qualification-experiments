echo ""
echo "--------------------------------------------------------------------------------"
echo "Running all experiments"
echo "--------------------------------------------------------------------------------"
echo ""

sh ./experiments/_run_straightline.sh navigation
sh ./experiments/_run_straightline.sh hvac
sh ./experiments/_run_drp.sh hvac
# sh ./experiments/_run_straightline.sh reservoir
# sh ./experiments/_run_drp.sh reservoir