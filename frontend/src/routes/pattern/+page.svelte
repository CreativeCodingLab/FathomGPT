<svelte:head>
    <title>Pattern Extractor</title>
    <meta name="description" content="Pattern Extractor" />
</svelte:head>

<script lang="ts">
    import { writable } from 'svelte/store';
    import { serverBaseURL } from '../../lib/Helpers/constants';
	import {Shadow} from 'svelte-loading-spinners';


    let uploadedImageUrl = writable('');
    let isImageUploaded = writable(false);
    let patternImagedata = writable('');
	let loading = false;

    function handleFilesChange(event: Event) {
        const input = event.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            uploadedImageUrl.set(URL.createObjectURL(input.files[0]));
            isImageUploaded.set(true);
        }
    }

    function clearImage() {
        uploadedImageUrl.set('');
        isImageUploaded.set(false);
    }

	async function handleImageClick(event: MouseEvent) {
    const imgElement = event.target as HTMLImageElement;

    if (imgElement) {
        const bounds = imgElement.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;

        const imageX = Math.round((x / imgElement.offsetWidth) * imgElement.naturalWidth);
        const imageY = Math.round((y / imgElement.offsetHeight) * imgElement.naturalHeight);

        const response = await fetch($uploadedImageUrl);
        const blob = await response.blob();

        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = function() {
            const base64data = reader.result;
			loading = true
            fetch(serverBaseURL+'/generate_pattern', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64data,
                    imageX: imageX,
                    imageY: imageY,
                }),
            })
            .then(response => response.json())
            .then(data => {
                patternImagedata.set(data.image);
				loading = false
            })
            .catch(error => {
				alert("Error while generating the pattern")
                console.error('Error:', error);
				loading = false
            });
        };
    }
}
</script>

<main>
    {#if $isImageUploaded}
        <div class="images-container">
            <div class="image-pane">
				<h5>Click on the image to generate the pattern</h5>
                <img src={$uploadedImageUrl} alt="Uploaded" on:click={handleImageClick}>
                <button on:click={clearImage} class="clear-button button">Clear</button>
            </div>
            <div class="image-pane">
				{#if $patternImagedata.length===0 && !loading}
                <p>Image is not clicked yet!</p>
				{:else if $patternImagedata.length!==0 }
				<img class="patternedImage" src={$patternImagedata} alt="Patterned">
				{/if}
				{#if loading}
				<div class="loadingContainer">
					<Shadow size="30" color="#FFFFFF" unit="px" duration="1s" />
				</div>
				{/if}
            </div>
        </div>
    {:else}
        <div class="upload-container">
            <input type="file" id="file-upload" accept="image/*" on:change={handleFilesChange}>
            <label for="file-upload" class="upload-button">Click to upload or drag and drop an image here</label>
        </div>
    {/if}
</main>

<style>
    main {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        flex-direction: column;
    }

    .upload-container, .images-container {
        width: 95%;
        max-width: 1200px;
        display: flex;
        justify-content: space-around;
    }

    .image-pane {
        flex-basis: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }

	.image-pane:first-child{
        border-right: 1px solid #ccc;
	}

	.image-pane:last-child{
        border-left: 1px solid #ccc;
		position: relative;
	}

	.patternedImage{
		padding-bottom: 43px;
    	padding-top: 6px;
	}

	.button{
		margin-top: 20px;
	}

	h5{
		margin-bottom: 10px;
	}

    img {
        max-width: 100%;
        max-height: 400px;
    }

    input[type="file"] {
        display: none;
    }

	.upload-button {
        display: inline-block;
        margin: 20px;
        padding: 10px 20px;
        background-color: #f0f0f0;
        border-radius: 5px;
        cursor: pointer;
    }

	.loadingContainer{
		background-color: rgba(51,51,51,0.5);
		position: absolute;
		width: 100%;
		height: 100%;
		display: flex;
		justify-content: center;
		align-items: center;
	}
</style>
