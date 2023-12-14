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

	let container: HTMLElement;
	let updateBox: Text;
	export let URL: string;
	export let guid: string | null = null;

	export async function fetchResponse(inputtedText: string) {

		//repeat user input into a user input text box component
		const userInput = new UserInput({ target: container, props: { text: inputtedText } });
		window.scrollTo(0, document.body.scrollHeight);

		let request = `${URL}?question=${inputtedText}`;
		if (guid !== null) {
			request += `&guid=${guid}`;
		}
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

	function handleResponse(eventData: any) {
		
		if(eventData.message != undefined || eventData.message != null) {
			if(updateBox != null) {
				//@ts-ignore
				updateBox.$destroy();
			}
			if(eventData.message.includes("Querying database")) {
				handleText(container, eventData.message);
			}
			//@ts-ignore
			updateBox = handleText(container, eventData.message);
		}
		if(eventData.result != undefined) {
			console.log(eventData.result)
			guid = eventData.result.guid
			//@ts-ignore
			updateBox.$destroy();
			console.log("Output type: ",eventData.result.outputType);
			switch (eventData.result.outputType) {
				case 'text':
					handleText(container, eventData.result.responseText);
					break;
				case 'image':
					handleImage(container, eventData.result);
					break;
				case 'histogram':
					console.log('histogram request');
				case 'heatmap':
					handleHeatMap(container, eventData.result);
					break;
				case 'species':
					console.log(eventData.responseText);
					handleTaxonomy(container, eventData.result);
					break;
				case 'vegaLite':
					handleVega(container, eventData.result);
					break;
				case 'table':
					handleTable(container, eventData.result);
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
		margin: 1rem auto;
		max-width: 900px;
		overflow: hidden;
		overscroll-behavior: contain;
	}
</style>
