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
	

	let container: EventResponseContainer;
	let promptBox: Prompt;

	function handleResponse(event: any) {
		promptBox.toggleLoading();
		console.log('response received', event.detail);
		container.fetchResponse(event.detail.value, event.detail.image);
	}

	function responseReceived() {
		promptBox.toggleLoading();
	}

</script>

<main>
	<Info on:submit={handleResponse}></Info>
	<div class="chatContainer">
		<EventResponseContainer {URL} bind:this={container} on:responseReceived={responseReceived} />
		<Prompt bind:this={promptBox} on:submit={handleResponse} />
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
    	overflow-x: hidden;

	}
</style>
