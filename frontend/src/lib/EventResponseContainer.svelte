<script lang="ts">
	import UserInput from '$lib/Responses/UserInput.svelte';
	import handleImage from '$lib/Handlers/ImageHandler';
	import handleText from '$lib/Handlers/TextHandler';
	import handleHeatMap from '$lib/Handlers/HeatMapHandler';
	import handleTaxonomy from '$lib/Handlers/TaxonomyHandler';
	import handleVega from '$lib/Handlers/VegaHandler';
	import { createEventDispatcher } from 'svelte';
	import { handleTable } from './Handlers/Tablehandler';
	const dispatch = createEventDispatcher();
	import { serverBaseURL } from './Helpers/constants'
	import { onMount } from 'svelte';
	import Loader from "./Loader.svelte"
	import { handleVisualization } from './Handlers/HandlePlotly';

	let container: HTMLElement;
	let updateBox: Text;
	let curLoader: Loader| null;
	export let URL: string;
	export let guid: string | null = null;

	export async function getImageDetails(id: string) {
		let request = `${URL}species-detail?id=${id}`;
		const eventSource = new EventSource(request);
		eventSource.addEventListener('message', (event: MessageEvent) => {
			let parsedData = JSON.parse(event.data);
			console.log('Received message:', parsedData);
			handleResponse(parsedData);
			if(parsedData.result != undefined) {
				console.log("Event Stream Closed");
				eventSource.close();
				dispatch( 'responseReceived');
			}
		});
	}

	export async function fetchResponse(inputtedText: string, inputtedImage: string) {

		//repeat user input into a user input text box component
		const userInput = new UserInput({ target: container, props: { text: inputtedText, image: inputtedImage } });
		window.scrollTo(0, document.body.scrollHeight);

		let request = `${URL}?question=${inputtedText}`;
		if (guid !== null) {
			request += `&guid=${guid}`;
		}
		
		if (inputtedImage != null) {
			const payload = {
				image: inputtedImage
			};
			const response = await fetch(`${serverBaseURL}/upload_image`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(payload)
			});

			if (!response.ok) {
				// If the response is not ok, log the error
				const errorData = await response.text();
				throw new Error(`Server returned status code ${response.status}: ${errorData}`);
			}

			const data = await response.json();
			if (data.guid) {
				console.log('Image uploaded successfully! GUID:', data.guid);
			} else {
				console.error('Failed to upload image:', data.error);
			}
			request += `&image=${data.guid}`;
		}
		const eventSource = new EventSource(request);
		eventSource.addEventListener('message', (event: MessageEvent) => {
			
			let parsedData = JSON.parse(event.data);
			console.log('Received message:', parsedData);
			let isScrollAtBottom = container.scrollHeight==Math.floor(container.getBoundingClientRect().height)
			console.log(isScrollAtBottom," isScrollAtBottom", container.scrollHeight, container.scrollTop, container.getBoundingClientRect().height)
			try{
				handleResponse(parsedData);
			} catch (e) {
				console.log(e)
				handleText(container, "<div style='color:red'>Error while sending the request or parsing the response</div>");
			}
			if(parsedData.result != undefined) {
				console.log("Event Stream Closed");
				eventSource.close();
				dispatch( 'responseReceived');
			}
			if(isScrollAtBottom){
				if (container.lastElementChild !== null) {
					container.lastElementChild.scrollIntoView({
						block: 'start',
						behavior: 'smooth',
					});
				}
			}


		});

			
		}


	function handleResponse(eventData: any) {
		if(eventData.message != undefined || eventData.message != null) {
			//if(updateBox != null) {
			//	//@ts-ignore
			//	updateBox.$destroy();
			//}
			//if(eventData.message.includes("Querying database")) {
			//	handleText(container, eventData.message);
			//}
			////@ts-ignore
			//updateBox = handleText(container, eventData.message);
			
			if(curLoader != null){
				//@ts-ignore
				curLoader.addStep(eventData.message);
			}
			else{
				curLoader = new Loader({
					target: container,
					props: {
						items:[eventData.message],
						progressStep: 0
					}
				})
			}
			
		}
		if(eventData.result != undefined) {
			if(curLoader!=null){
				//@ts-ignore
				curLoader.complete()
				curLoader = null;
			}
			guid = eventData.result.guid
			//@ts-ignore
			//updateBox.$destroy();
			console.log("Output type: ",eventData.result.outputType);

			switch (eventData.result.outputType) {
				case 'text':
					handleText(container, eventData.result.responseText);
					break;
				case 'images':
					if(eventData.result.species){
						for(let i=0;i<eventData.result.species.length;i++){
							if(eventData.result.species[i].bb2!=null){
								eventData.result.species[i].id = eventData.result.species[i].bb2
							}
						}
						if(eventData.result.species.length>0&&eventData.result.species[0].CosineSimilarity!=null){
							eventData.result.species = eventData.result.species.sort((a:any,b:any)=>b.CosineSimilarity-a.CosineSimilarity)
						}
						handleImage(container, eventData.result);
					}
					else{
						handleText(container, "No such images found in the database.");
					}
					break;
				case 'histogram':
					console.log('histogram request');
				case 'heatmap':
					handleHeatMap(container, eventData.result);
					break;
				case 'taxonomy':
					handleTaxonomy(container, eventData.result);
					break;
				case 'vegaLite':
					handleVega(container, eventData.result);
					break;
				case 'table':
					handleTable(container, eventData.result);
					break;
				case 'visualization':
					handleVisualization(container, eventData.result);
					break;
				case 'error':
					handleText(container, "<div style='color:red'>"+eventData.result.responseText+"</div>");
					break;
				default:
					console.error('[TREY] Error: Invalid output type');
					handleText(container, 'Error: Invalid output type');
					break;
			}
		}
		//scroll to bottom of the page
		//window.scrollTo(0, document.body.scrollHeight);
	}
</script>

<main bind:this={container} />

<style>
	main {
		min-height: 100dvh;
		display: flex;
		flex-flow: column;
		justify-content: flex-end;
		gap: 1rem;
		align-self: center;
		max-width: 1000px;
		margin: 10px var(--page-horizontal-padding);
		overflow: hidden;
		overscroll-behavior: contain;
		width: 100%;
		padding-top: 10px;
		padding-left: 20px;
		padding-right: 20px;
	}


	main > :global(.fathomChatContainer){
		border-radius: var(--chat-bubble-radius);
		padding: var(--chat-padding);
		width: 100%;
		background-color: white;
	}
</style>
