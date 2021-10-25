# For the first time 
# sudo chmod a+x run.sh


this_path=$( dirname $(realpath "$0") )
path_data="/home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/_Data/"

#Activate virtualenv
source /home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/05-venvs/jovenes-talento/bin/activate

#Run Code
cd $this_path
python3 main.py --dataroot "$path_data"