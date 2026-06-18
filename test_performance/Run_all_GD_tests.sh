# Batch runner to generate ground truth.
#
# It executes generates_ground_truth.py over all the text cases.

source ~/venv-firedrake/bin/activate


cd ~/Desktop/tesi/tesi

echo "test case 1"
python src/maintomove.py test_performance/test_case1/settings_test1.1.json
python src/maintomove.py test_performance/test_case1/settings_test1.2.json
python src/maintomove.py test_performance/test_case1/settings_test1.3.json


echo "test case 2"
python src/maintomove.py test_performance/test_case2/settings_test2.json


echo "test case 3"
python src/maintomove.py test_performance/test_case3/settings_test3.json