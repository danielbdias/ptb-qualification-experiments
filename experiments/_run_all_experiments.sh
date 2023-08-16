echo ""
echo "--------------------------------------------------------------------------------"
echo "Running all experiments"
echo "--------------------------------------------------------------------------------"
echo ""

start=$(date +%s)

sh ./_run_drp.sh reservoir
sh ./_run_drp.sh hvac
sh ./_run_straightline.sh reservoir
sh ./_run_straightline.sh hvac
sh ./_run_straightline.sh navigation

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"