<script lang="ts">
	import { onMount } from 'svelte';
	import vegaEmbed, { type VisualizationSpec } from 'vega-embed';

	export let spec: VisualizationSpec;
	export let responseText: string;

	let visual: HTMLDivElement;

	onMount(() => {
		vegaEmbed(visual, spec);
		window.scrollTo(0, document.body.scrollHeight);
	});
</script>

<main>
	<h3>Fathom said:</h3>
	<br/>
	<p>{responseText}</p>
	<div bind:this={visual} />
</main>

<style>
	main {
		width: 100%;
		border-radius: 0.5rem;
		padding: 0.5rem;
		background-color: var(--background-light);
		min-height: 5rem;
		color: white;
		display: grid;
		place-items: center start;
	}
</style>
