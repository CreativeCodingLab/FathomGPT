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
    

def stream(request):
    guid = request.data.get('guid')
    new_question = request.data.get('question')

    main_object = None
    new_answer = None
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

    #Interaction.objects.create(main_object=main_object, request=new_question, response=new_answer)

    response = StreamingHttpResponse(run_prompt(new_question, messages, True), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'  # Important for streaming to work properly
    return response