<svelte:head>
	<title>Ovai GPT</title>
	<meta name="description" content="OVAI GPT Prototype" />
</svelte:head>

<script lang="ts">
	import Info from '$lib/Info.svelte';
	import Prompt from '$lib/Prompt.svelte';
	import EventResponseContainer from '$lib/EventResponseContainer.svelte';

	

	const URL = 'http://128.46.81.243:8000/';
	//const URL = 'http://127.0.0.1:8000/event-stream';
	

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
