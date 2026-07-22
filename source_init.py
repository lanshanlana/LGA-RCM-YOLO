#1.find ultralytics path
import ultralytics
import os
import shutil
#1.find ultralytics path
wk_dir = os.getcwd()
ul_path = ultralytics.__path__[0]
ul_nn_path = os.path.join(ul_path,"nn")
#2.copy Attention folder to uu_nn_path
lga_and_rcm_path = os.path.join(wk_dir,"lga_rcm_yolo")
attention_path = os.path.join(ul_nn_path,"Attention")
if not os.path.exists(attention_path):
    os.mkdir(attention_path)
shutil.copytree(os.path.join(lga_and_rcm_path,"Attention"),attention_path,dirs_exist_ok=True)
#3 replace modules.block.py and __init__.py with the new one
new_block_folder_path = os.path.join(lga_and_rcm_path,"modules")
shutil.copytree(new_block_folder_path,os.path.join(ul_nn_path,"modules"),dirs_exist_ok=True)
#4.replace tasks.py with the new one
new_tasks_path = os.path.join(lga_and_rcm_path,"tasks.py")
shutil.copy(new_tasks_path,os.path.join(ul_nn_path,"tasks.py"))
print("All files have been replaced successfully!")
