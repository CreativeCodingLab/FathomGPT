from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import MainObject, Interaction, Image
from .serializers import MainObjectSerializer, InteractionSerializer
from .llm import run_promptv1
from .interact_thre import patternDivision
from .segment import segment
from django.http import JsonResponse
import sys
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
sys.path.insert(0, '..')
from llm.main import run_prompt
from django.views.decorators.csrf import csrf_exempt
from django.http import StreamingHttpResponse
import time
import json
from django.views import View
import pymssql
import os
import uuid
from django.http import HttpResponse
import base64
from llm.langchaintools import getTaxonomyTree
import cv2
import numpy as np

sqlServer = os.getenv("SQL_SERVER")
database = os.getenv("DATABASE")
dbUser = os.getenv("DB_USER")
dbPwd = os.getenv("DB_PWD")

class MainObjectViewSet(viewsets.ModelViewSet):
    queryset = MainObject.objects.all()
    serializer_class = MainObjectSerializer


    @csrf_exempt
    @action(detail=False, methods=['POST'])
    def create_with_question_answer(self, request):
        question = request.data.get('question')

        if question is None:
            return Response({"error": "Please add a question"}, status = 400)
        answer = run_prompt(question, [])

        main_object = MainObject.objects.create()
        Interaction.objects.create(main_object=main_object, request=question, response=answer)

        return Response({'guid': str(main_object.id), 'response': answer}, status=status.HTTP_201_CREATED)

    @csrf_exempt
    @action(detail=False, methods=['PUT'])
    def update_with_question_answer(self, request):
        guid = request.data.get('guid')
        new_question = request.data.get('question')

        main_object=None
        try:
            main_object = MainObject.objects.get(id=guid)
        except MainObject.DoesNotExist:
            return Response({'status': 'GUID not found'}, status=status.HTTP_404_NOT_FOUND)


        messages = []


        for interaction in main_object.interactions.all():
            question = interaction.request
            answer = interaction.response

            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        new_answer = run_prompt(new_question, messages)
        
        Interaction.objects.create(main_object=main_object, request=new_question, response=new_answer)

        return Response({'guid': str(main_object.id), 'response': new_answer}, status=status.HTTP_200_OK)
    
    @csrf_exempt
    @action(detail=False, methods=['GET'])
    def getSpeciesDetail(self, request):
        species_id = None
        if(request.GET.get('id') != None):
            species_id = request.GET.get('id')
        else:
            species_id = request.GET.get('bounding_box_id')

        if species_id is None or species_id == "":
            return Response({},status=status.HTTP_400_BAD_REQUEST)

        connection = pymssql.connect(
            server=sqlServer,
            user=dbUser,
            password=dbPwd,
            database=database
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT bb.*, img.*, CASE WHEN img.latitude IS NOT NULL AND img.longitude IS NOT NULL THEN mr.name ELSE 'No Region Data Available' END AS region_name FROM dbo.bounding_boxes AS bb INNER JOIN dbo.images AS img ON bb.image_id = img.id LEFT JOIN dbo.marine_regions AS mr ON (img.latitude BETWEEN mr.min_latitude AND mr.max_latitude) AND (img.longitude BETWEEN mr.min_longitude AND mr.max_longitude) WHERE bb.id = "+species_id+" FOR JSON AUTO;")

        rows = cursor.fetchall()

        content = ''.join(str(row[0]) for row in rows)
        cursor.close()
        connection.close()
        print("content", content)
        parsed = json.loads(content)[0]
        taxonomy=json.loads(getTaxonomyTree(parsed['concept']))
        parsed['rank'] = taxonomy['rank']
        parsed['taxonomy'] = taxonomy['taxonomy']

        return Response(parsed, status=status.HTTP_200_OK)

    @csrf_exempt
    @action(detail=False, methods=['POST'])
    def upload_image(self, request):
        if request.method == 'POST':
            data = json.loads(request.body)
            image_data = data['image']
            format, imgstr = image_data.split(';base64,') 
            ext = format.split('/')[-1] 

            file_name = str(uuid.uuid4()) + '.' + ext
            image_file = ContentFile(base64.b64decode(imgstr), name=file_name)

            image_instance = Image.objects.create(
                guid=uuid.uuid4(),
                image=image_file
            )
            
            return JsonResponse({'guid': str(image_instance.guid)})
        else:
            return JsonResponse({'error': 'Invalid request'}, status=400)

    @csrf_exempt
    @action(detail=False, methods=['POST'])
    def segment_image(self, request):
        if request.method == 'POST':
            data = request.data
            image_base64 = data['image'].split(",")[1]
            imageX = data['imageX']
            imageY = data['imageY']
            
            img_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            processed_img = segment(imageX, imageY, img)
            
            img_base64 = []

            _, buffer1 = cv2.imencode('.jpg', processed_img[0])
            _, buffer2 = cv2.imencode('.jpg', processed_img[1])
            _, buffer3 = cv2.imencode('.jpg', processed_img[2])
            img_base64.append(base64.b64encode(buffer1).decode())
            img_base64.append(base64.b64encode(buffer2).decode())
            img_base64.append(base64.b64encode(buffer3).decode())
            
            return JsonResponse({'image0': 'data:image/jpeg;base64,' + img_base64[0], 'image1': 'data:image/jpeg;base64,' + img_base64[1], 'image2': 'data:image/jpeg;base64,' + img_base64[2]})
        
    @csrf_exempt
    @action(detail=False, methods=['POST'])
    def generate_pattern(self, request):
        if request.method == 'POST':
            data = request.data
            image_base64 = data['image'].split(",")[1]
            imageX = data['imageX']
            imageY = data['imageY']
            color_thre = data['color_thre']
            
            img_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            processed_img = patternDivision(imageX, imageY, img, color_thre)
            
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_base64 = base64.b64encode(buffer).decode()
            
            return JsonResponse({'image': 'data:image/jpeg;base64,' + img_base64})


    

def event_stream(new_question, image, messages, isEventStream, db_obj):
    while True:
        yield from run_prompt(new_question, image, messages, isEventStream=isEventStream, db_obj = db_obj)
        time.sleep(10)
        break

class PostStreamView(View):

    def get(self, request):
        guid = request.GET.get('guid')
        new_question = request.GET.get('question')
        imageguid = request.GET.get('image')
        base64_image = ''

        if imageguid is not None:
            try:
                # Retrieve the image object from the database
                image_obj = Image.objects.get(guid=imageguid)
                
                # Open the image file associated with the image object
                with open(image_obj.image.path, "rb") as image_file:
                    # Read the file content and encode it in Base64
                    base64_encoded_image = base64.b64encode(image_file.read())
                    # Decode the Base64 bytes object into a string
                    base64_image = base64_encoded_image.decode('utf-8')
                    
            except Image.DoesNotExist:
                # Handle the case where no image is found for the provided imageguid
                return HttpResponse('Image not found', status=404)
            except Exception as e:
                # Handle other potential errors
                return HttpResponse(str(e), status=500)


        main_object = None
        messages = []
        if(guid != None):
            try:
                main_object = MainObject.objects.get(id=guid)
            except MainObject.DoesNotExist:
                return Response({'status': 'GUID not found'}, status=status.HTTP_404_NOT_FOUND)

            for interaction in main_object.interactions.all():
                question = interaction.request
                answer = interaction.response

                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        else:
            main_object = MainObject.objects.create()

        #

        print("generating response")
        response = StreamingHttpResponse(event_stream(new_question, base64_image, messages, True, main_object))#
        response['Cache-Control'] = 'no-cache'
        response['Content-Type'] = 'text/event-stream'
        return response

