<script lang="ts">
	import type { speciesData } from '$lib/types/responseType';
	import { activeImageStore } from '../../store';

	//export let speciesData: speciesData[];
	export let species: Array<speciesData>;
	export let naturalTextResponse: string;
	function openImageDetailModal(specimen: speciesData) {
		activeImageStore.set({ isImageDetailsOpen: true, species: specimen });
	}
</script>

<main class="fathomChatContainer">
	<blockquote>{naturalTextResponse}</blockquote>
	<br />
	<div>
		{#each species as specimen, i}
			<button class="imgOuterWrapper" on:click={()=>openImageDetailModal(specimen)}>
				<!-- svelte-ignore a11y-img-redundant-alt -->
				<div class="imgWrapper">
					<img src={specimen.url} alt="image retrieved from fathomnet"/>
				</div>
				<h4>Name: {specimen.concept}</h4>
				{#if specimen.CosineSimilarity!=null}
				<h4>Similarity Score: {(specimen.CosineSimilarity * 100).toFixed(2)}%</h4>
				{/if}
			</button>
		{/each}
	</div>
</main>

<style>
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
		transition: 0.2s ease;
		cursor: pointer;
	}
	.imgOuterWrapper:hover img{
		transform: scale(1.1);
	}
	.imgWrapper{
		overflow: hidden;
		border-radius: 0.5rem;

	}
	.imgOuterWrapper{
		border: 0;
		width: unset;
		height: unset;
		background: transparent;
		text-align: left;
		color: white;
	}
	.imgOuterWrapper h4{
		margin-top: 3px;
	}

	.imgOuterWrapper:hover h4{
		text-decoration: underline;
		cursor: pointer;
	}

	.imgOuterWrapper:active{
		opacity: 0.9;
	}
</style>
