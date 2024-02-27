import kfp
from kfp import dsl

@dsl.component
def preprocess_data_op(container_image: str):
    return dsl.ContainerOp(
        name='Data Augmentation',
        image=container_image,
        arguments=[],
        file_outputs={
            'output': '/augmented_data',  # Path inside the container where output will be saved
        }
    ).set_memory_limit('2G').set_cpu_limit('1')

@dsl.pipeline(
    name='Data Augmentation Pipeline',
    description='A pipeline that augments data.'
)
def data_preprocessing_pipeline():
    preprocess_task = preprocess_data_op(container_image='us-east1-docker.pkg.dev/ta-model-data-preprocess/ta-model-pipelines/data_augmentation_test_1:test')

# Compile the pipeline
kfp.compiler.Compiler().compile(pipeline_func=data_preprocessing_pipeline, package_path='data_preprocessing_pipeline.yaml')
