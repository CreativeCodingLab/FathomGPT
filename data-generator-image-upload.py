import csv
import json
import random

instructions = {
    "text": "Make sure the response text is a templated string so that data can be formatted inside the text",
    "images": "The sql query must have bounding box id of the species, concept of the species and the image url of the species on all inputs",
    "imagesWithInput": "The prompt will asks for similar images, there is another system that takes in the similarImageIDs and similarBoundingBoxIDs that you generated above to calculate the similarity search. You will suppose the similarity search is already done and you have sql table SimilaritySearch that has the input bounding box id as bb1, output bounding box id as bb2 and Cosine Similarity Score as CosineSimilarity. You will use this table and add the conditions that is given provided by the user. You will also ouput the ouput bounding box image url and the concept. The result must be ordered in descending order using the CosineSimilarity value. Also, you will take 10 top results unless specified by the prompt",
    "visualization": "Generate sample data and corresponding Plotly code.Guarantee that the produced SQL query and Plotly code are free of syntax errors and do not contain comments.In the Plotly code, ensure all double quotation marks ("") are properly escaped with a backslash ().Represent newline characters as \\n and tab characters as \\t within the Plotly code",
    "table": "The response text can be templated so that it can hold the count of the data array from the sql query result.",
}

# Open the TSV file and read its contents
with open('SQL Query and Response 2.txt', mode='r', newline='', encoding='utf-8') as file:
    # Create a CSV reader for a tab-separated values file
    reader = csv.DictReader(file, delimiter='\t')

    data = []
    for row in reader:
        row["GPT Response"] = json.loads(row["GPT Response"])
        data.append(row)
    
    # Open the output file in write mode
    with open('output_with_plotly_and_image_features.jsonl', 'w', encoding='utf-8') as outfile:
        
        # Iterate through the rows and write each row as a JSON object on a new line in the output file
        for index, row in enumerate(data):
            curResponse = json.loads(json.dumps(row["GPT Response"]))
            if row['InputImageAvailable'] != "TRUE":
                continue

            filteredRes = [x for x in data if x["GPT Response"]['outputType'] == curResponse['outputType'] and x["GPT Response"]['prompt'] != curResponse['prompt'] and x['InputImageAvailable'] == row['InputImageAvailable']]

            oneShotPrompt = random.choice(filteredRes)
            oneShotPromptResponse = json.loads(json.dumps(oneShotPrompt["GPT Response"]))

            oneShotPromptText = oneShotPromptResponse['prompt']
            del oneShotPromptResponse['prompt']
            del oneShotPromptResponse['outputType']
            del oneShotPromptResponse['findings']
            if 'plotlyCode' in oneShotPromptResponse:
                oneShotPromptResponse['plotlyCode'] = oneShotPromptResponse['plotlyCode'].replace("    ","\t")

            curResPrompt = curResponse['prompt']
            del curResponse['prompt']
            outputType = curResponse['outputType']
            del curResponse['outputType']
            del curResponse['findings']
            if 'plotlyCode' in curResponse:
                curResponse['plotlyCode'] =  curResponse['plotlyCode'].replace("    ","\t")
            #       "sampleData": "",
            #        "plotlyCode": "",
            #    sampleData: This is the sample data that you think should be generated when running the sql query. This is optional. It is only needed when the outputType is visualization
            #    plotlyCode: This is the python plotly code that you will generate. You will generate a function named "drawVisualization(data)". The function should take in data variable. The data value will have the structur of the sampleData generated above. Donot redfine the sample data here. The code should have the necessary imports and the "drawVisualization" function. This attribute is optional but must be generated only when the outputType is visualization.
                 
            messages = [{
"role": "system", "content": """You are a very intelligent json generated that can generate highly efficient sql queries. You will be given an input prompt for which you need to generated the JSON in a format given below, nothing else.
The Generated SQL must be valid for Micorsoft sql server
The JSON format and the attributes on the JSON are provided below
{
 "similarImageIDs": [],
 "similarBoundingBoxIDs": [],
 "similarImageSearch": true/false,
 "sqlServerQuery": "",
 "responseText": ""
}
similarImageIDs: these are the image id that will be provided by the user in the prompt on which image search needs to be done
similarBoundingBoxIDs: these are the bounding_boxes id that will be provided by the user in the prompt on which bounding boxes search needs to be done
similarImageSearch: this is a boolean field, that is true when the prompt says to find similar images, else it is false
sqlServerQuery: This is the sql server query you need to generate based on the user's prompt. The database structure provided will be very useful to generate the sql query. 
responseText: Suppose you are answering the user with the output from the prompt. You need to write the message in this section. When the response is text, you need to output the textResponse in a way the values from the generated sql can be formatted in the text

"""+ (instructions['imagesWithInput'] if (oneShotPrompt['InputImageAvailable'] == "TRUE" or row['InputImageAvailable'] == "TRUE") else "") + "\n"+
                instructions[outputType]
            },
            #{
            #    "role": "user","content": f"""
            #    User Prompt: {oneShotPromptText}
            #    Output type: {outputType}
            #    InputImageDataAvailable: {oneShotPrompt['InputImageAvailable']}"""
            #},
            #{
            #   # "role": "assistant", "content": "Observation:"+row["Observation"]+"\nJSON:```"+row["Sql query"]+"```"
            #    "role": "assistant", "content": json.dumps(oneShotPromptResponse)
            #},
            {
                "role": "user","content": f"""
Microsoft SQL Server Database Structure:

CREATE TABLE "dbo"."bounding_box_comments" ( "id" bigint NOT NULL, "bounding_box_uuid" uniqueidentifier NULL, "created_timestamp" datetime2(6) NULL, "email" varchar(254) NULL, "last_updated_timestamp" datetime2(6) NULL, "text" varchar(2048) NULL, "uuid" uniqueidentifier NOT NULL, "alternate_concept" varchar(255) NULL, "flagged" bit NULL, CONSTRAINT "PK__bounding__3213E83F71625CCD" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."bounding_boxes" ("id" bigint NOT NULL,"concept" varchar(255) NULL,"created_timestamp" datetime2(6) NULL,"group_of" bit NULL,"height" int NULL,"last_updated_timestamp" datetime2(6) NULL,"observer" varchar(256) NULL,"occluded" bit NULL,"truncated" bit NULL,"uuid" uniqueidentifier NOT NULL,"verification_timestamp" datetimeoffset(6) NULL,"verified" bit NULL,"verifier" varchar(256) NULL,"width" int NULL,"x" int NULL,"y" int NULL,"image_id" bigint NULL,"alt_concept" varchar(255) NULL,"user_defined_key" varchar(56) NULL, "magnitude" decimal(18,5) NULL, CONSTRAINT "PK__bounding__3213E83F3E4C2D08" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE dbo.bounding_box_image_feature_vectors ( bounding_box_id bigint NOT NULL, vector_index int NULL, vector_value decimal(18,5) NULL, CONSTRAINT FK_bounding_box_image_feature_vectors_bounding_box_id FOREIGN KEY (bounding_box_id) REFERENCES dbo.bounding_boxes (id) ON DELETE CASCADE);
CREATE TABLE "dbo"."bounding_boxes_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "concept" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "group_of" bit NULL, "height" int NULL, "last_updated_timestamp" datetime2(6) NULL, "observer" varchar(256) NULL, "occluded" bit NULL, "truncated" bit NULL, "uuid" uniqueidentifier NULL, "verification_timestamp" datetimeoffset(6) NULL, "verified" bit NULL, "verifier" varchar(256) NULL, "width" int NULL, "x" int NULL, "y" int NULL, "image_id" bigint NULL, "alt_concept" varchar(255) NULL, "user_defined_key" varchar(56) NULL, CONSTRAINT "PK__bounding__BE3894F99D30F28A" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."darwin_cores" ( "id" bigint NOT NULL, "access_rights" varchar(1024) NULL, "basis_of_record" varchar(64) NULL, "bibliographic_citation" varchar(512) NULL, "collection_code" varchar(64) NULL, "collection_id" varchar(2048) NULL, "data_generalizations" varchar(512) NULL, "dataset_id" uniqueidentifier NULL, "dataset_name" varchar(255) NULL, "dynamic_properties"varchar(2048) NULL, "information_withheld" varchar(255) NULL, "institution_code" varchar(255) NULL, "institution_id" varchar(255) NULL, "license" varchar(2048) NULL, "modified" datetimeoffset(6) NULL, "owner_institution_code" varchar(255) NULL, "record_language" varchar(35) NULL, "record_references" varchar(2048) NULL, "record_type" varchar(32) NULL, "rights_holder" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "image_set_upload_id" bigint NULL, CONSTRAINT "PK__darwin_c__3213E83F92DAE497" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."fathomnet_identities" ( "id" bigint NOT NULL, "api_key" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "disabled" bit NULL, "display_name" varchar(255) NULL, "email" varchar(255) NULL, "expertise_rank" varchar(32) NULL, "firebase_uid" varchar(255) NULL, "job_title" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "organization" varchar(255) NULL, "profile" varchar(1024) NULL, "role_data" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "avatar_url" varchar(2000) NULL, "orcid" varchar(32) NULL, "notification_frequency" varchar(32) NULL, CONSTRAINT "PK__fathomne__3213E83F59FE1468" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."fathomnet_identities_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "api_key" varchar(255) NULL, "avatar_url" varchar(2000) NULL, "created_timestamp" datetime2(6) NULL, "disabled" bit NULL, "display_name" varchar(255) NULL, "email" varchar(254) NULL, "expertise_rank" varchar(32) NULL, "firebase_uid" varchar(64) NULL, "job_title" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "orcid" varchar(32) NULL, "organization" varchar(255) NULL, "profile" varchar(1024) NULL, "role_data" varchar(255) NULL, "uuid" uniqueidentifier NULL, "notification_frequency" varchar(32) NULL, CONSTRAINT "PK__fathomne__BE3894F9CD98EF78" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."followed_topics" ( "id" bigint NOT NULL, "created_timestamp" datetime2(6) NULL, "email" varchar(254) NULL, "last_updated_timestamp" datetime2(6) NULL, "notification" bit NULL, "target" varchar(256) NULL, "topic" varchar(32) NULL, "uuid" uniqueidentifier NOT NULL, CONSTRAINT "PK__followed__3213E83F4A1EA9E0" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."image_set_uploads" ( "id" bigint NOT NULL, "contributors_email" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "format" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "local_path" varchar(2048) NULL, "rejection_details" varchar(255) NULL, "rejection_reason" varchar(255) NULL, "remote_uri" varchar(2048) NULL, "sha256" varchar(64) NULL, "status" varchar(255) NULL, "status_update_timestamp" datetimeoffset(6) NULL, "status_updater_email" varchar(254) NULL, "uuid" uniqueidentifier NOT NULL, "darwincore_id" bigint NULL, CONSTRAINT "PK__image_se__3213E83F9C72A0E9" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."image_uploads_join" ( "imagesetupload_id" bigint NOT NULL, "image_id" bigint NOT NULL, CONSTRAINT "PK__image_up__8A53EE0EBD9D776A" PRIMARY KEY CLUSTERED("imagesetupload_id","image_id")
ON [PRIMARY]);
CREATE TABLE "dbo"."images" ( "id" bigint NOT NULL, "created_timestamp" datetime2(6) NULL, "depth_meters" float NULL, "height" int NULL, "imaging_type" varchar(64) NULL, "last_updated_timestamp" datetime2(6) NULL, "last_validation" datetimeoffset(6) NULL, "latitude" float NULL, "longitude" float NULL, "media_type" varchar(255) NULL, "modified" datetimeoffset(6) NULL, "sha256" varchar(64) NULL, "submitter" varchar(255) NULL, "timestamp" datetimeoffset(6) NULL, "url" varchar(2048) NULL, "uuid" uniqueidentifier NOT NULL, "valid" bit NULL, "width" int NULL, "contributors_email"varchar(254) NULL, "altitude" float NULL, "oxygen_ml_l" float NULL, "pressure_dbar" float NULL, "salinity" float NULL, "temperature_celsius" float NULL, CONSTRAINT "PK__images__3213E83FD76E309F" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."images_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "contributors_email"varchar(254) NULL, "created_timestamp" datetime2(6) NULL, "depth_meters" float NULL, "height" int NULL, "imaging_type" varchar(64) NULL, "last_updated_timestamp" datetime2(6) NULL, "last_validation" datetimeoffset(6) NULL, "latitude" float NULL, "longitude" float NULL, "media_type" varchar(255) NULL, "modified" datetimeoffset(6) NULL, "sha256" varchar(64) NULL, "timestamp" datetimeoffset(6) NULL, "url" varchar(255) NULL, "uuid" uniqueidentifier NULL, "valid" bit NULL, "width" int NULL, "altitude" float NULL, "oxygen_ml_l" float NULL, "pressure_dbar" float NULL, "salinity" float NULL, "temperature_celsius" float NULL, CONSTRAINT "PK__images_a__BE3894F92DF6715C" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."marine_regions" ( "id" bigint NOT NULL, "created_timestamp" datetime2 NULL, "last_updated_timestamp" datetime2 NULL, "max_latitude" float NOT NULL, "max_longitude" float NOT NULL, "min_latitude" float NOT NULL, "min_longitude" float NOT NULL, "mrgid" bigint NULL, "name" varchar(255) NULL, CONSTRAINT "PK__marine_r__3213E83FE987FC8B" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."revinfo" ( "rev" int IDENTITY(1,1) NOT NULL, "revtstmp" bigint NULL, CONSTRAINT "PK__revinfo__C2B7CC69D3938648" PRIMARY KEY CLUSTERED("rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."tags" ( "id" bigint NOT NULL, "created_timestamp" datetime2 NULL, "tag" varchar(255) NOT NULL, "last_updated_timestamp" datetime2 NULL, "media_type" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "value" varchar(255) NOT NULL, "image_id" bigint NULL, CONSTRAINT "PK__tags__3213E83F3A490684" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."tags_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "created_timestamp" datetime2 NULL, "tag" varchar(255) NULL, "last_updated_timestamp" datetime2 NULL, "media_type" varchar(255) NULL, "uuid" uniqueidentifier NULL, "value" varchar(255) NULL, "image_id" bigint NULL, CONSTRAINT "PK__tags_aud__BE3894F94F1608A8" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE VIEW "dbo"."boundingbox_extended_info" AS
SELECT b.concept, b.alt_concept, b.observer, b.verified, b.verifier, b.verification_timestamp, b.user_defined_key, i.url, i.width, i.height, i.submitter, i.[timestamp], i.contributors_email AS image_contributors_email, u.contributors_email AS upload_contributors_email, dc.owner_institution_code, dc.institution_code, dc.rights_holder, dc.collection_code, dc.collection_id, dc.dataset_name
FROM dbo.darwin_cores dc LEFT JOIN dbo.image_set_uploads u ON dc.id = u.darwincore_id LEFT JOIN dbo.image_uploads_join j ON j.imagesetupload_id = u.id LEFT JOIN dbo.images i ON j.image_id = i.id LEFT JOIN dbo.bounding_boxes b ON b.image_id = i.id;


                User Prompt: {curResPrompt}
                """
            },
            # 
            {
               # "role": "assistant", "content": "Observation:"+row["Observation"]+"\nJSON:```"+row["Sql query"]+"```"
                "role": "assistant", "content": json.dumps(curResponse)
            }]
            filtered_row = {
                "messages": messages
            }
            
            # Convert the dictionary to a JSON string and write it to the file
            json_str = json.dumps(filtered_row)
            outfile.write(json_str + '\n')  # Add a newline character after each JSON object

print('Conversion completed successfully!')
