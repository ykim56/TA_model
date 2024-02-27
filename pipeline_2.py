import kfp
from kfp.dsl import pipeline, component
from google.cloud import aiplatform

# Load the component from the YAML definition
no_input_data_processing_component = kfp.components.load_component_from_text(
    open("data_processing_component.yaml", "r").read()
)



@component(
    base_image="us-east1-docker.pkg.dev/ta-model-data-preprocess/ta-model-pipelines/data_augmentation_test_1:test",
)
def set_machine_type_op():
    return aiplatform.ModelDeployOp(
        project='ta-model-data-preprocess',
        location='us-east1',
        #machine_type='n1-standard-4'  # Specify your desired machine type
    )

# Load the component from the YAML definition
data_processing_component = kfp.components.load_component_from_text(open("data_processing_component.yaml", "r").read())



# Define the pipeline
@pipeline(
    name='Data Processing Pipeline',
    description='A pipeline that processes data using a custom Docker image.'
)
def data_pipeline():
    # Set the machine type
    #machine_type_op = set_machine_type_op()

    # Use the component in the pipeline
    data_processing = data_processing_component().set_cpu_limit('1').set_memory_limit('4G')
    #data_processing.after(machine_type_op)

# Compile the pipeline
kfp.compiler.Compiler().compile(data_pipeline, 'data_pipeline.yaml')
