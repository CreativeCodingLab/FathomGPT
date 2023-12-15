from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import MainObject, Interaction
from .serializers import MainObjectSerializer, InteractionSerializer
from .llm import run_promptv1
import sys
sys.path.insert(0, '..')
from llm.main import run_prompt
from django.views.decorators.csrf import csrf_exempt
from django.http import StreamingHttpResponse
import time
import json
from django.views import View
import pymssql
import os

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
        species_id = request.GET.get('id')
        if species_id is None or species_id == "":
            return Response({},status=status.HTTP_400_BAD_REQUEST)

        connection = pymssql.connect(
            server=sqlServer,
            user=dbUser,
            password=dbPwd,
            database=database
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT bb.*, img.* FROM dbo.bounding_boxes AS bb INNER JOIN dbo.images AS img ON bb.image_id = img.id WHERE bb.id = "+species_id+" FOR JSON AUTO;")

        rows = cursor.fetchall()

        content = ''.join(str(row[0]) for row in rows)
        cursor.close()
        connection.close()
        print(type(content))

        return Response(json.loads(content), status=status.HTTP_200_OK)



    

def event_stream(new_question, messages, isEventStream, db_obj):
    while True:
        yield from run_prompt(new_question, messages, isEventStream=isEventStream, db_obj = db_obj)
        time.sleep(10)
        break

class PostStreamView(View):

    def get(self, request):
        guid = request.GET.get('guid')
        new_question = request.GET.get('question')


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
        response = StreamingHttpResponse(event_stream(new_question, messages, True, main_object))#
        response['Cache-Control'] = 'no-cache'
        response['Content-Type'] = 'text/event-stream'
        return response

