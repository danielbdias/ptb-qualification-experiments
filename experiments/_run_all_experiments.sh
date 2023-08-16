echo ""
echo "--------------------------------------------------------------------------------"
echo "Running all experiments"
echo "--------------------------------------------------------------------------------"
echo ""

sh ./_run_drp.sh reservoir
sh ./_run_drp.sh hvac
sh ./_run_straightline.sh reservoir
sh ./_run_straightline.sh hvac
sh ./_run_straightline.sh navigation