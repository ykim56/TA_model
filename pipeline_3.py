# IMPORT REQUIRED LIBRARIES
from kfp import dsl
from kfp.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        Markdown,
                        HTML,
                        component, 
                        OutputPath, 
                        InputPath)
from kfp import compiler
from google.cloud.aiplatform import pipeline_jobs
import pandas as pd
from datetime import datetime

REGION = 'us-east1'
BUCKET_NAME = 'ta-charts-data'
PROJECT_NAME = 'TA-model-data-preprocess'
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/"
BASE_IMAGE = f"us-east1-docker.pkg.dev/ta-model-data-preprocess/ta-model-pipelines/data_augmentation_test_2:test"


# 'A pipeline that performs augmentation'
# DeprecationWarning: output_component_file parameter is deprecated and will eventually be removed. Please use `Compiler().compile()` to compile a component instead.
#@component(base_image=BASE_IMAGE, output_component_file="augment_data.yaml")  
@component(base_image=BASE_IMAGE)   
def augment_data(
    filepath: str,
    start_date: str,
    end_date: str,
    augmented_data: Output[Dataset],
):
    import pandas as pd
    from utils.augment_price_data import augment_price_data, obtain_syn_data, labeling_function, tech_indicator_calc
    
    df = pd.read_csv("gs://" + filepath + '/*.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    spy_df_synth = obtain_syn_data(original_df=df, lam=0.5, start_date=start_date, end_date=end_date)
    spy_df_synth.to_csv(augmented_data.path, index=False)

    
# descriptions='labeling buy, sell, no-action signals'    
#@component(base_image=BASE_IMAGE, output_component_file="labeled_data.yaml")
@component(base_image=BASE_IMAGE)
def label_data(
    df: Input[Dataset],
    label_name: str,
    labeled_data: Output[Dataset],
):
    from utils.augment_price_data import augment_price_data, obtain_syn_data, labeling_function, tech_indicator_calc       
    
    df = pd.read_csv(df.path)
    df_labeled = labeling_function(df, label_name=label_name)
    df_labeled.to_csv(labeled_data.path, index=False)


# descriptions='obtain technical indicators'
#@component(base_image=BASE_IMAGE, output_component_file="data_with_ta.yaml")
@component(base_image=BASE_IMAGE)
def obtain_ta(
    df: Input[Dataset],
    data_with_ta: Output[Dataset],
):
    from utils.augment_price_data import augment_price_data, obtain_syn_data, labeling_function, tech_indicator_calc   
    
    df = pd.read_csv(df.path)
    df_with_ta = tech_indicator_calc(df)
    df_with_ta.to_csv(data_with_ta.path, index=False)
    
    

# USE TIMESTAMP TO DEFINE UNIQUE PIPELINE NAMES
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = 'data-pipeline-job{}'.format(TIMESTAMP)
    

@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="data-preprocess-pipeline"   
)
def pipeline(
    data_filepath: str = f"{BUCKET_NAME}/origin_data",
    project: str = PROJECT_NAME,
    region: str = REGION, 
    display_name: str = DISPLAY_NAME,
    start_date: str = '2000-01-04',
    end_date: str = '2023-12-22'
):
    augment_data_op = augment_data(filepath=data_filepath, start_date=start_date, end_date=end_date)
    augment_data_op.set_cpu_limit('1').set_memory_limit('4G')  #.set_machine_type('n1-standard-1')
    
    
    
    #data_op = get_houseprice_data(data_filepath)
    #data_preprocess_op = preprocess_houseprice_data(data_op.outputs["dataset_train"])
    #train_test_split_op = train_test_split(data_preprocess_op.outputs["dataset_train_preprocessed"])
    #train_model_op = train_houseprice(train_test_split_op.outputs["dataset_train"], train_test_split_op.outputs["dataset_test"])
    #model_evaluation_op = evaluate_houseprice(train_model_op.outputs["model"])
    

# COMPILE THE PIPELINE (to create the job spec file)
compiler.Compiler().compile(pipeline_func=pipeline, package_path='data_pipeline.yaml')