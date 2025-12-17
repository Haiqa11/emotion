import runpy
import os

this_dir = os.path.dirname(__file__)
app_path = os.path.join(this_dir, 'ser_ravdess_6class', 'app.py')
runpy.run_path(app_path, run_name='__main__')