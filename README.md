#Create Virtual Environment
conda create -p venv1 python==3.12 -y
#activate venv1
conda activate venv1/ 
or create a new repository on the command line
echo "# Student_marks" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Himansh9532/Student_marks.git
>git remote -v
git push -u origin main
#add 
git add .
git status
git commit 
git push
# create .gitignore file 
# Create Requiremnets.txt
# Create setup.py file 
# Create SRC FILE  
1.components
a-> data injection
b -> data transformation
c-> model_trainer.py
d-> model_trainer.py
2.pipeline
a->
predict_pipeline.py
b->train_piprline.py
# 2._INIT__.py
# 3.logger.py ->>>>>>>>>>>>>>>>>> python src/logger.py
 # 4. exception.py
 # 5. utils.py
# run requirements.txt file 

create logfile 
create exception file -> python src/exception.py
go to data_ingestion and write code in data ingestion load , train , test data write tthe code after that run terminal ->>>>>> python src/components/data_ingestion.py
<!-- and check artifacts it create or not# -->
write code in data transformation  and save this file 
and write code in utils to save file 
after that run code data ingestion and check it runs properly 