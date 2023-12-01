<script lang="ts">
	import Prompt from '$lib/Prompt.svelte';
	import ResponseContainer from '$lib/ResponseContainer.svelte';

	const URL = 'http://128.46.81.243:8000/get_response';

	let container: ResponseContainer;
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
	<ResponseContainer {URL} bind:this={container} on:responseReceived={responseReceived} />
	<Prompt bind:this={promptBox} on:submit={handleResponse} />
</main>

<style>
	main {
		width: 99vw;
		height: 100dvh;
		overscroll-behavior: none;
	}
</style>
