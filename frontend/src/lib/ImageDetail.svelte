<script lang="ts">
    import { activeImageStore } from '../store';
    import type { speciesData } from '$lib/types/responseType';
    import { browser } from '$app/environment';
    import { serverBaseURL } from './Helpers/constants';
    import { derived } from 'svelte/store';
    import { formattedDate } from './Helpers/formatters';

    const isOpen = derived(activeImageStore, $store => $store.isImageDetailsOpen);
    $: species = $activeImageStore.species;
    $: if ($isOpen) handleOpenChange();

    function handleOpenChange() {
        if(!browser)
            return
        document.body.style.overflow = $isOpen ? 'hidden' : '';
        if($isOpen){
            boundingBoxStyle=""
            fetch(serverBaseURL+"/species_detail?id="+species?.id)  .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                let newSpeciesData: speciesData = {...data.img[0], ...data};
                activeImageStore.update(current => ({ ...current, species: newSpeciesData }));

                let setBoundingBoxStyle = function(){
                    if(newSpeciesData.x!==undefined && newSpeciesData.y!==undefined && newSpeciesData.width!==undefined && newSpeciesData.height!==undefined){
                        boundingBoxStyle = calculateBoundingBoxStyle(newSpeciesData.x,newSpeciesData.y,newSpeciesData.width,newSpeciesData.height)
                    }
                }
                if (imgElement.complete) {
                    setBoundingBoxStyle()
                } else {
                    imgElement.addEventListener('load', setBoundingBoxStyle);
                }
            })
            .catch(error => {
                alert(`Error fetching species detail`);

            });
        }
    }

    let imgElement: HTMLImageElement;
    let boundingBoxStyle: string = '';


    function calculateBoundingBoxStyle(x:number, y:number, width:number, height:number) {
        const xPercent = (x / imgElement.naturalWidth) * 100;
        const yPercent = (y / imgElement.naturalHeight) * 100;
        const widthPercent = (width / imgElement.naturalWidth) * 100;
        const heightPercent = (height / imgElement.naturalHeight) * 100;

        const style = `display: block; left: ${xPercent}%; top: ${yPercent}%; width: ${widthPercent}%; height: ${heightPercent}%;`;

        return style;
    }


    function closeModal() {
        activeImageStore.update(current => ({ ...current, isImageDetailsOpen: !current.isImageDetailsOpen }));
    }
</script>
<main>
	<div class="imageDetailsOuterContainer" class:active={$isOpen}>
        {#if species!==null}
		<div class="imageDetailsContainer">
            <div class="header">
                <h1>{species.concept}</h1>
                <button class="closeBtn" on:click={closeModal}><i class="fa-solid fa-xmark"></i></button>
            </div>
            <div class="imageContainer">
                <img src="{species.url}" alt="{species.concept}" bind:this={imgElement}/>
                <div class="boundBoxContainer" style={boundingBoxStyle}>
                </div>
            </div>
			<div class="detailsContainer">
				<div class="detailsTextContainer">
					<h3>Details</h3>
                    {#if species.mr!=null}
					    <div>Region: <b>{species.mr[0].region_name}</b></div>
                    {/if}
                    {#if species.created_timestamp!=null}
                        <div>Created: <b>{formattedDate(species.created_timestamp)}</b></div>
                    {/if}
                    {#if species.last_updated_timestamp!=null}
                        <div>Last Updated: <b>{formattedDate(species.last_updated_timestamp)}</b></div>
                    {/if}
                    {#if species.last_updated_timestamp != null}
                    <div>Salinity Level: <b>{species.salinity} ppt</b></div>
                    {/if}
                    {#if species.oxygen_ml_l != null}
                        <div>Oxygen Level: <b>{species.oxygen_ml_l} ml/L</b></div>
                    {/if}
                    {#if species.temperature_celsius != null}
                        <div>Temperature: <b>{species.temperature_celsius}&deg;C</b></div>
                    {/if}
                    {#if species.pressure_dbar != null}
                        <div>Pressure level: <b>{species.pressure_dbar} dbar</b></div>
                    {/if}
				</div>
				<div class="taxonomyContainer">
					<h3>Taxonomy</h3>
                    {#if species.taxonomy != null}
                    <div class="innerTaxonomyContainer">
                        {#each species.taxonomy?.ancestors as ancestor, index}
                        <div class="taxonomyItem" style="{'margin-left:'+(index*12)+"px"}"><b>{ancestor.name}</b> ({ancestor.rank})</div>
                        {/each}
                        <div class="taxonomyItem" style="{'margin-left:'+(species.taxonomy?.ancestors.length*12)+"px"}"><b>{species.concept}</b> ({species.rank})</div>
                        {#each species.taxonomy?.descendants as descendant, index}
                        <div class="taxonomyItem" style="{'margin-left:'+((species.taxonomy?.ancestors.length+index)*12)+"px"}"><b>{descendant.name}</b> ({descendant.rank})</div>
                        {/each}
                    </div>
                    {/if}
                </div>
			</div>
		</div>
        {/if}
	</div>
</main>

<style>
    .imageDetailsOuterContainer{
        position: fixed;
        left: 0;
        top: 0;
        width: 100%;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        background-color: rgba(0, 0, 0, 0.5);
        opacity: 0;
        pointer-events: none;
        transition: 0.2s;
        backdrop-filter: blur(1px);
    }
    .imageDetailsOuterContainer.active{
        opacity: 1;
        pointer-events: all;
    }
	.imageDetailsContainer {
		padding: 20px;
        max-width: 1200px;
        min-width: 600px;
        margin: 20px;
        border-radius: 10px;
        background: white;
        display: flex;
        flex-direction: column;
	}
    .detailsContainer{
        display: flex;
        flex-direction: row;
        padding-top: 5px;
    }
    .detailsTextContainer{
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    .detailsTextContainer h3{
        padding-bottom: 3px;
    }
    .taxonomyContainer{
        
    }
    .innerTaxonomyContainer{
        padding-left: 2px;
    }
    .taxonomyItem:not(:first-child){
        position: relative;
    }
    .taxonomyItem:not(:first-child)::after{
        position: absolute;
        width: 5px;
        height: 12px;
        border-left: 2px solid rgba(0, 0, 0, 0.3);
        border-bottom: 2px solid rgba(0, 0, 0, 0.3);
        display: block;
        content: '';
        top: -3px;
        left: -8px;
    }
    h1{
        padding: 5px 0px;
    }
    .detailsTextContainer div{
        color: #111;
        padding: 3px 0px;
    }
    .detailsTextContainer div:first-child{
        padding-top: 5px;
    }

    .closeBtn{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 0;
        margin-bottom: 10px;
        background: rgba(0, 0, 0, 0.1);
        cursor: pointer;
        align-self: flex-end;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .closeBtn:hover{
        background: rgba(0, 0, 0, 0.2);
    }
    .closeBtn:active{
        background: rgba(0, 0, 0, 0.3);
    }
    .imageDetailsContainer .header{
        display: flex;
        justify-content: space-between;
    }
    .imageDetailsContainer .imageContainer{
        padding: 5px 0px;
        position: relative;
    }
    .imageContainer img{
        width: 100%;
    }
    .imageContainer .boundBoxContainer{
        position: absolute;
        display: none;
        border: 3px solid rgb(0, 141, 242);
        background-color: rgba(0, 141, 242, 0.2);
    
    }
</style>
