<script lang="ts">
    import { createEventDispatcher } from "svelte";
	import { onMount } from 'svelte';
    let sidebarHidden:Boolean=true
	let backgroundRef: Element;

    const dispatch = createEventDispatcher();

    function insertPrompt(event: any) {
        sidebarHidden=true
        dispatch("submit", {value:event.target.innerText});
    }
    let prompts = [
        "Find me images of Aurelia aurita",
        "Find me images of moon jelly sorted by depth",
        "Find me best images of moon jelly",
        "Find me images of creatures with tentacles in Monterey bay and depth less than 5k meters",
        "Find me images of starfish in Monterey bay and depth less than 5k meters",
        "Find me best images of ray-finned creatures in Monterey bay and depth less than 5k meters",
        "Find me images of predators of moon jelly",
        "Find me images of orange creatures",
        "What are the ancestors of moon jelly",
        "Show me the taxonomy of Aurelia aurita",
        "What color are moon jellyfsh",
        "How large are moon jellyfish",
        "Generate an Interactive Time-lapse Map of Marine Species Observations Grouped by Year",
        "Generate an area chart showing the year the images were taken",
        "Display a bar chart showing the distribution of all species in Monterey Bay, categorized by the salinity level they are found in",
        "What is the total number of Patiria miniata found in the database?",
        "Generate a heatmap of all species in Monterey Bay",
        "Generate a scatterplot between oxygen and pressure data in Monetery bay from the database",
        "Display a pie chart that correlates salinity levels with the distribution of Bathochordaeus stygius categorizing salinity levels from 30 to 38 with each level of width 1",
        "Display a bar chart showing the temperature ranges for Aurelia Aurita and Pycnopodia helianthoides colored coded for each species from 0°C to 20 in 5°C increments.",
        "Generate a box plot for the showing the oxygen level at which Aurelia Aurita live",
        //"Group bar chart showing the pressure level for Bathochordaeus stygius and Aurelia Aurita categorized in below 2, 2 to 10, 10 to 100, 100 to 1000, above 1000 pressure levels",
        "Find me any 3 species of starfish and get me their total image count in the database",
    ]

    function reload(){
        window.location.reload()
    }

    function hideSidebar(event: any){
        if(event.target === backgroundRef){
            sidebarHidden = true
        }
    }

    onMount(() => {
        document.addEventListener('sidebarEvents', function (e) {
            if(e.detail.sidebarOpened){
                sidebarHidden = false
            }
        });
	})
</script>
<main on:click={hideSidebar} class:sidebarHidden="{sidebarHidden}" bind:this={backgroundRef}>
    <div class="sidebar">
        <div class="stickyContainer">
            <button class="closeSidebarBtn buttonCircled"><i class="fa-solid fa-xmark" on:click={()=>sidebarHidden = true}></i></button>
            <button class="button newBtn" on:click={reload}> <i class="fa-solid fa-plus"></i> New chat</button>
            <h2>Try Asking:</h2>
            <hr>
            <sub>Click on a prompt to insert it into the input box</sub>
        </div>
        <ul>
            {#each prompts as prompt}
                <!-- svelte-ignore a11y-click-events-have-key-events -->
                <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
                <li on:click={insertPrompt}>{prompt}</li>
            {/each}
        </ul>
    </div>
</main>

<style>
    main {
        /*position this element to the top left corner of the page when scrolling down*/
        width: 25rem;
        background-color: var(--color-white);
        overflow-y: scroll;
        color: black;
        padding: 0px 30px 30px var(--page-horizontal-padding);
		height: calc(100vh - var(--page-header-height));
        position: sticky;
        top: var(--page-header-height);
		transition: 0.2s ease;
		@media (max-width: 1080px) {
			position: fixed;
            top: 0;
            z-index: 10000;
            height: 100vh;
            width: 100%;
            background-color: rgba(51,51,51,0.3);
            padding-left: 0px;
            padding-bottom: 0px;
            overflow-x: hidden;
		}
    }

    main.sidebarHidden{
		@media (max-width: 1080px) {
            background-color: transparent;
            pointer-events: none;
            overflow: hidden;
        }
    }

    .sidebar{
		@media (max-width: 1080px) {
            width: 25rem;
            background-color: white;
            padding-left: 20px;
            padding-right: 20px;
            padding-bottom: 20px;
            padding-top: 20px;
            transition: 0.2s ease;
        }
    }

    main.sidebarHidden .sidebar{
		@media (max-width: 1080px) {
            transform: translateX(-100%);
        }
    }

    .newBtn{
		@media (max-width: 1080px) {
            display: none;
        }
    }

    sub{
        color: var(--color-pelagic-gray);
    }

    .stickyContainer{
        position: sticky;
        top: 0px;
        background-color: var(--color-white);
        padding-top: 30px;
        padding-bottom: 16px;
        z-index: 10;
        width: calc(100% + 10px);
		@media (max-width: 1080px) {
            padding-top: 10px;
        }
    }

    h2 {
        font-size: 1.5rem;
        font-weight: 400;
        padding-top: 20px;
    }
    ul {
        list-style: none;
    }
    li {
        font-size: 1rem;
        font-weight: 300;
        padding: 0 0 1.2rem 0;
        transition-duration: 0.1s;
        cursor: pointer;
    }

    li:hover {
        color: var(--accent-color);
        transform: scale(1.05);
        transform: translate(0.5rem, 0);
    }

    .closeSidebarBtn{
        display: none;
		@media (max-width: 1080px) {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    }

    .closeSidebarBtn i{
        font-size: 24px;
    }

</style>