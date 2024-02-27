from google.cloud import aiplatform

project_id = 'ta-model-data-preprocess'
region = 'us-east1'
aiplatform.init(project=project_id, location=region)

pipeline_path = 'data_pipeline.yaml'
pipeline_display_name = 'data_augmentation_pipeline'

pipeline_job = aiplatform.PipelineJob(
    display_name=pipeline_display_name,
    template_path=pipeline_path,
    enable_caching=False,
    location=region
    # parameter_values={'param1': 'value1', 'param2': 'value2'}
)

#pipeline_job.run()
pipeline_job.submit()