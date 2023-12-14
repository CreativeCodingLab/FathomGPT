<script lang="ts">
	import type { speciesData } from '$lib/types/responseType';

	//export let speciesData: speciesData[];
	export let concepts: string[];
	export let imageArray: string[];
	export let naturalTextResponse: string;
</script>

<main>
	<h3>Fathom said:</h3>
	<blockquote>{naturalTextResponse}</blockquote>
	<div>
		{#each imageArray as entry, i}
			<div>
				<!-- svelte-ignore a11y-img-redundant-alt -->
				<img src={entry} alt="image retrieved from fathomnet"/>
				<h4>Name: {concepts[i]}</h4>
			</div>
		{/each}
	</div>
</main>

<style>
	main {
		width: 100%;
		border-radius: 0.5rem;
		padding: 0.5rem;
		background-color: var(--background-light);
		min-height: 3rem;
		color: white;
		display: grid;
		place-items: center start;
	}
	div {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
		grid-template-rows: repeat(auto-fill, minmax(10rem, 1fr));
		width: 100%;
		gap: 0.5rem;
		height: fit-content;
	}

	img {
		width: 100%;
		height: 100%;
		border-radius: 0.5rem;
	}
</style>
