<script lang="ts">
    export let generate_html: string;
    export let naturalTextResponse: string;
    import { onMount } from 'svelte';

    let curVisContainer: HTMLDivElement

    onMount(() => {
        console.log(curVisContainer)
        const visualizationData = JSON.parse(generate_html);
        Plotly.newPlot(curVisContainer, visualizationData.data, visualizationData.layout);
    })
</script>

<main class="fathomChatContainer">
	<blockquote>{naturalTextResponse}</blockquote>
	<br />
	<div class="plotLyVis" bind:this={curVisContainer}>
	</div>
</main>

<style>
		.plotLyVis{
        min-width: 250px;
        min-height: 450px;
		}
</style>
