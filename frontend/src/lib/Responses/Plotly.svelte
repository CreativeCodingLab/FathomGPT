<script lang="ts">
    export let generate_html: string;
    export let naturalTextResponse: string;
    import { onMount } from 'svelte';
	import { tick } from 'svelte';


    let curVisContainer: HTMLDivElement

    onMount(async () => {
        curVisContainer.innerHTML+=generate_html
        let scriptParent = curVisContainer.querySelector("script")?.parentElement
        let script = curVisContainer.querySelector("script")
        if(script!=null && scriptParent!=null){
            scriptParent?.removeChild(script)
            let newScript = document.createElement('script');
            newScript.type="text/javascript"
            newScript.innerHTML = script.innerHTML;
            document.body.appendChild(newScript)
        }


        //const visualizationData = JSON.parse(generate_html);
        //Plotly.newPlot(curVisContainer, visualizationData.data, visualizationData.layout);
    })
</script>

<main class="fathomChatContainer">
	<blockquote>{naturalTextResponse}</blockquote>
	<br />
    <div class="plotlyWrapper">
	    <div class="plotLyVis" bind:this={curVisContainer}>
    </div>
	</div>
</main>

<style>
		.plotlyWrapper{
            width: calc(100vw - 80px);
            max-width: 920px;
            overflow: auto;

		}

        .plotLyVis{
            min-width: 550px;
            min-height: 450px;
        }
</style>
