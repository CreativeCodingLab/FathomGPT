<svelte:head>
	<title>Ovai GPT</title>
	<meta name="description" content="OVAI GPT Prototype" />
</svelte:head>

<script lang="ts">
	import Info from '$lib/Info.svelte';
	import Prompt from '$lib/Prompt.svelte';
	import EventResponseContainer from '$lib/EventResponseContainer.svelte';
	import ImageDetail from '$lib/ImageDetail.svelte';
	import { serverBaseURL } from '$lib/Helpers/constants';

	const URL = serverBaseURL+'/event-stream';
	

	let container: EventResponseContainer;
	let promptBox: Prompt;

	function handleResponse(event: any) {
		promptBox.toggleLoading();
		console.log('response received', event.detail);
		container.fetchResponse(event.detail);
	}

	function responseReceived() {
		promptBox.toggleLoading();
	}

</script>

<main>
	<Info on:submit={handleResponse}></Info>
		<EventResponseContainer {URL} bind:this={container} on:responseReceived={responseReceived} />
		<Prompt bind:this={promptBox} on:submit={handleResponse} />
		<ImageDetail></ImageDetail>
</main>

<style>
	main {
		margin: 0 auto;
		padding-left: 20rem;
		max-width: 70vw;
		height: 100dvh;
		overscroll-behavior: none;
	}
</style>
