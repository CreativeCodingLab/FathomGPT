<svelte:head>
	<title>Ovai GPT</title>
	<meta name="description" content="OVAI GPT" />
</svelte:head>

<script lang="ts">
	import Info from '$lib/Info.svelte';
	import Prompt from '$lib/Prompt.svelte';
	import EventResponseContainer from '$lib/EventResponseContainer.svelte';
	import ImageDetail from '$lib/ImageDetail.svelte';
	import { serverBaseURL } from '$lib/Helpers/constants';
	import ImageUploader from '$lib/Components/ImageUploader.svelte';

	const URL = serverBaseURL+'/event-stream';
	let isNewChatNeeded = false
	

	let container: EventResponseContainer;
	let promptBox: Prompt;

	function handleResponse(event: any) {
		if(isNewChatNeeded){
			return
		}
		promptBox.toggleLoading();
		container.fetchResponse(event.detail.value, event.detail.image);
	}

	function addNewRequest(event: any) {
		promptBox.addNewRequest(event.detail.value, event.detail.image)
	}

	function responseReceived() {
		promptBox.toggleLoading();
	}

	function setNewChatNeeded() {
		isNewChatNeeded = true
	}

</script>

<main>
	<Info on:submit={addNewRequest}></Info>
	<div class="chatContainer">
		<EventResponseContainer {URL} bind:this={container} on:responseReceived={responseReceived} on:newChatNeeded={setNewChatNeeded} />
		<Prompt bind:this={promptBox} on:submit={handleResponse} isNewChatNeeded={isNewChatNeeded} />
	</div>
	<ImageDetail></ImageDetail>
	<ImageUploader></ImageUploader>
</main>

<style>
	main {
		margin: 0 auto;
		min-height: 100dvh;
		display: flex;
    	margin-top: var(--page-header-height);
	}

	.chatContainer{
		display: flex;
		flex-direction: column;
		flex: 1;
		background-color: var(--color-sea-salt-gray);
		max-width: 100vw;
	}
</style>
