import json
import os
import importlib
import subprocess
import pandas as pd
import shutil
from library import DefaultClassificationTemplate
from dsbox.datapreprocessing.cleaner import config as cleaner_config
a = DefaultClassificationTemplate()

def get_meta_json(dataset_name):
    # generate the meta file for pipelines
    # if pipeline_type == "classification":
    #     dataset_name = "38_sick"
    # elif pipeline_type == "regression":
    #     dataset_name = "196_autoMpg"
    # # more pipeline types needed

    meta_json = {
                "problem": dataset_name + "_problem",
                "full_inputs": [dataset_name + "_dataset"],
                "train_inputs": [dataset_name + "_dataset_TRAIN"],
                "test_inputs": [dataset_name + "_dataset_TEST"],
                "score_inputs": [dataset_name + "_dataset_SCORE"]
            }
    
    return meta_json

def get_primitive_hitted(config):
    """
        Return a list of DSBOX primitives that are found in the config file
        We should only add sample pipelines for our own primitives
    """
    primitive_hitted = []
    for each_primitive in config.values():
        temp = each_primitive['primitive']
        if temp[-5:] == "DSBOX":
            primitive_hitted.append(temp)
    return primitive_hitted

def generate_pipeline(config:dict, meta_json):
    """
        Generate sample pipelines and corresponding meta
    """
    primitive_hitted = get_primitive_hitted(config)
    for each_primitive in primitive_hitted:
        outdir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
                              cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
                              cleaner_config.VERSION, "pipelines")
        os.makedirs(outdir, exist_ok=True)
        print("Generating at " + outdir + pipeline_json['id'] + "...")
        try:
            # generate the new pipeline
            pipeline = a.to_pipeline(config)
            pipeline_json = pipeline.to_json_structure()
            file_name = os.path.join(outdir, pipeline_json['id']+".json")
            meta_name = os.path.join(outdir, pipeline_json['id']+".meta")
            with open(file_name,"w") as f:
                json.dump(pipeline_json,f,separators=(',', ':'),indent=4)
            with open(meta_name,"w") as f:
                json.dump(meta_json,f,separators=(',', ':'),indent=4)
            print("succeeded!")
        except:
            print("!!!!!!!")
            print("failed!")
            print("!!!!!!!")

def remove_temp_files():
    tmp_files = os.listdir("tmp")
    for each_file in tmp_files:
        file_path = os.path.join("tmp", each_file)
        os.remove(file_path)

def clean_up_primitive_repo():
    try:
        dataset_ids = os.listdir("dsbox-unit-test-datasets")
        for each in dataset_ids:
            if each[0] != ".":
            shutil.rmtree("dsbox-unit-test-datasets" + each)
        shutil.move("output", "dsbox-unit-test-datasets")
    except:
        print("clean up failed!")

def test_pipeline(each_config_name, config, test_dataset_id):
    try:
        pipeline = a.to_pipeline(config)
        pipeline_json = pipeline.to_json_structure()
        temp_pipeline = os.path.join("tmp/test_pipeline.json")
        with open(temp_pipeline,"w") as f:
            json.dump(pipeline_json,f)
        d3m_runtime_command = "python -m d3m.runtime -d dsbox-unit-test-datasets fit-produce -p tmp/test_pipeline.json -r dsbox-unit-test-datasets/" + \
                              test_dataset_id + "/TRAIN/problem_TRAIN/problemDoc.json -i dsbox-unit-test-datasets/" + \
                              test_dataset_id + "/TRAIN/dataset_TRAIN/datasetDoc.json -t dsbox-unit-test-datasets/" + \
                              test_dataset_id + "/TEST/dataset_TEST/datasetDoc.json -o tmp/produced_output.csv" 


        p = subprocess.Popen(d3m_runtime_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        p.wait()
        try:
            # load prediction file
            predictions = pd.read_csv("tmp/produced_output.csv")
        except:
            print("predictions file load failed, please check the pipeline.")
            return False

        # load ground truth file
        ground_truth = pd.read_csv("dsbox-unit-test-datasets/"+test_dataset_id+"/mitll_predictions.csv")

        if predictions.columns.all() != ground_truth.columns.all():
            print("prediction columns are:")
            print(predictions.columns)
            print("ground truth columns are:")
            print(ground_truth.columns)
            print("The predictions columns and ground truth columns are not same.")
            return False

        if set(predictions['d3mIndex']) != set(ground_truth['d3mIndex']):
            temp1 = list(set(predictions['d3mIndex']))
            temp1.sort()
            print("prediction indexes are:")
            print(temp1)
            temp2 = list(set(ground_truth['d3mIndex']))
            temp2.sort()
            print("ground truth indexes are:")
            print(temp2)
            print("The prediction d3mIndex and ground truth d3mIndex are not same.")
            return False

        return True
    except:
        raise ValueError("Running train-test with config" + each_config_name +"failed!")
        return False

def main():
    # clean up the old output files if necessary
    try:
        shutil.rmtree("output")
    except:
        pass

    config_list = os.listdir("pipeline_configs")
    # generate pipelines for each configuration
    for each_config_name in config_list:
        # only check python files
        if each_config_name[-2:] == "py":
            each_config_name = each_config_name[:-3]
            imp_module = "pipeline_configs." + each_config_name
            ip_module = importlib.import_module('.', imp_module)
            ip_module_cls = getattr(ip_module, 'config')
            cls_obj = ip_module_cls()
            config = cls_obj.config
            pipeline_type = cls_obj.pipeline_type
            test_dataset_id = cls_obj.test_dataset_id

            meta_json = get_meta_json(test_dataset_id)
            result = test_pipeline(each_config_name, config, test_dataset_id)
            remove_temp_files()
            # only generate the pipelines with it pass the test
            if result:
                print("Test pipeline passed! Now generating the pipeline json files...")
                generate_pipeline(config, meta_json)
            else:
                print("Test pipeline not passed! Please check the detail errors")
                raise ValueError("Auto generating pipelines failed")

    clean_up_primitive_repo()
if __name__ == "__main__":
    main()

