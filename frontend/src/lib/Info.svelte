<script lang="ts">
    import { createEventDispatcher } from "svelte";
	import { onMount } from 'svelte';
    let sidebarHidden:Boolean=true
	let backgroundRef: Element;

    const dispatch = createEventDispatcher();

    function insertPrompt(event: any) {
        hideSideBar()
        dispatch("submit", {value:event.target.innerText});
    }
    let prompts = [
        "What color are moon jellyfish",
        "What is the scientific name of moon jelly",
        "What are some creatures with caudal fins",

        "Find me images of Aurelia aurita",
        "Find me images of moon jelly sorted by depth",
        "Find me images of moon jelly in Monterey bay and depth less than 5k meters",
        "Find me best images of moon jelly",

        "Find me images of creatures with caudal fins",
        "Find me images of orange creatures",
        "Find me images of predators of moon jelly",
        "Find me images of creatures found in tropical seas",

        "What are the ancestors of moon jelly",
        "Show me the taxonomy of Aurelia aurita",

        "Find me any 3 species of starfish and get me their total image count in the database",
        "Generate me a list of top 20 species from the database with their count",


        "Generate an area chart showing the year the images were taken",
        "Display a bar chart showing the distribution of all species in Monterey Bay, categorized by the salinity level they are found in",
        "What is the total number of Patiria miniata found in the database?",
        "Display a heatmap of all species in Monterey Bay",
        "Generate a scatterplot between oxygen and pressure data in Monetery bay from the database",
        "Generate a pie chart showing the count of Bathochordaeus stygius across 7 salinity ranges from 30 to 38, each range spanning 1 unit.",
        "Produce a bar chart displaying temperature ranges (0°C to 20°C) for Aurelia Aurita and Pycnopodia helianthoides, each species color-coded.",
        "Generate a box-plot showing the oxygen levels at which images that have Aurelia Aurita live.",
        //"Group bar chart showing the pressure level for Bathochordaeus stygius and Aurelia Aurita categorized in below 2, 2 to 10, 10 to 100, 100 to 1000, above 1000 pressure levels",
    ]

    function reload(){
        window.location.reload()
    }

    function hideSideBar(){
        sidebarHidden = true
		document.body.style="overflow:auto"
    }

    function showSideBar(){
		document.body.style="overflow:hidden"
        sidebarHidden = false
    }

    function hideSidebarEvent(event: any){
        if(event.target === backgroundRef){
            hideSideBar()
        }
    }

    onMount(() => {
        document.addEventListener('sidebarEvents', function (e) {
            if(e.detail.sidebarOpened){
                showSideBar()
            }
        });
	})
</script>
<main on:click={hideSidebarEvent} class:sidebarHidden="{sidebarHidden}" bind:this={backgroundRef}>
    <div class="sidebar">
        <div class="stickyContainer">
            <button class="closeSidebarBtn buttonCircled"><i class="fa-solid fa-xmark" on:click={hideSideBar}></i></button>
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
		height: calc(100dvh - var(--page-header-height));
        position: sticky;
        top: var(--page-header-height);
		transition: 0.2s ease;
		@media (max-width: 1080px) {
			position: fixed;
            top: 0;
            z-index: 10000;
            height: 100dvh;
            width: 100%;
            background-color: rgba(51,51,51,0.4);
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