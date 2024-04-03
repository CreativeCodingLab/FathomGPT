<script lang="ts">
	import { writable } from 'svelte/store';
	import { serverBaseURL } from '../../lib/Helpers/constants';
	import { Shadow } from 'svelte-loading-spinners';

	let uploadedImageUrl = writable('');
	let isImageUploaded = writable(false);
	let patternImagedata0 = writable('');
	let patternImagedata1 = writable('');
	let patternImagedata2 = writable('');
	let loading = false;

	let selectedPatternImage = writable('');
	let extractedPatternData = writable('');
	let isPatternExtract = writable(false);
	let color_thre = '1';

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

	function backToImageMaskSelect() {
		isPatternExtract.set(false);
	}

	/// if we click one of the masked image, move on to pattern extraction
	function handleImageMaskSelect(number: number) {
		//TODO
		switch (number) {
			case 0:
				isPatternExtract.set(true);
				selectedPatternImage.set($patternImagedata0);
				break;
			case 1:
				isPatternExtract.set(true);
				selectedPatternImage.set($patternImagedata1);
				break;
			case 2:
				isPatternExtract.set(true);
				selectedPatternImage.set($patternImagedata2);
				break;
			default:
				break;
		}
		imageCropping();
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
			reader.onloadend = function () {
				const base64data = reader.result;
				loading = true;
				fetch(serverBaseURL + '/segment_image', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						image: base64data,
						imageX: imageX,
						imageY: imageY
					})
				})
					.then((response) => response.json())
					.then((data) => {
						patternImagedata0.set(data.image0);
						patternImagedata1.set(data.image1);
						patternImagedata2.set(data.image2);
						loading = false;
					})
					.catch((error) => {
						alert('Error while generating the pattern');
						console.error('Error:', error);
						loading = false;
					});
			};
		}
	}

	async function generatePattern(event: MouseEvent) {
		const imgElement = event.target as HTMLImageElement;

		if (imgElement) {
			const bounds = imgElement.getBoundingClientRect();
			const x = event.clientX - bounds.left;
			const y = event.clientY - bounds.top;

			const imageX = Math.round((x / imgElement.offsetWidth) * imgElement.naturalWidth);
			const imageY = Math.round((y / imgElement.offsetHeight) * imgElement.naturalHeight);

			// const response = await fetch($uploadedImageUrl);
			// const blob = await response.blob();

			// const reader = new FileReader();
			// reader.readAsDataURL(blob);
			// reader.onloadend = function () {
			// const base64data = reader.result;
			const base64data = $selectedPatternImage;
			loading = true;
			fetch(serverBaseURL + '/generate_pattern', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					image: base64data,
					imageX: imageX,
					imageY: imageY,
					color_thre: Number(color_thre)
				})
			})
				.then((response) => response.json())
				.then((data) => {
					extractedPatternData.set(data.image);
					loading = false;
				})
				.catch((error) => {
					alert('Error while generating the pattern');
					console.error('Error:', error);
					loading = false;
				});
			// };
		}
	}

	async function imageCropping() {
			const base64data = $selectedPatternImage;
			loading = true;
			fetch(serverBaseURL + '/crop_image', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					image: base64data,
				})
			})
				.then((response) => response.json())
				.then((data) => {
					selectedPatternImage.set(data.image);
					loading = false;
				})
				.catch((error) => {
					alert('Error while generating the pattern');
					console.error('Error:', error);
					loading = false;
				});
	}

</script>

<svelte:head>
	<title>Pattern Extractor</title>
	<meta name="description" content="Pattern Extractor" />
</svelte:head>

<main>
	{#if $isImageUploaded}
		{#if !$isPatternExtract}
			<div class="images-container">
				<div class="image-pane firstImagePane">
					<h5>Click on the image to select the fish</h5>
					<img src={$uploadedImageUrl} alt="Uploaded" on:click={handleImageClick} />
					<button on:click={clearImage} class="clear-button button">Clear</button>
				</div>
                <div class="outputWrapper">
                    <div class="image-pane">
                        {#if $patternImagedata0.length === 0 && !loading}
                            <p>Image is not clicked yet!</p>
                        {:else if $patternImagedata0.length !== 0}
                            <img class="patternedImage" src={$patternImagedata0} alt="Patterned" />
                            <button class="button" on:click={() => handleImageMaskSelect(0)}>Select</button>
                        {/if}
                    </div>
                    <div class="image-pane">
                        {#if $patternImagedata1.length === 0 && !loading}
                            <p>Image is not clicked yet!</p>
                        {:else if $patternImagedata1.length !== 0}
                            <img class="patternedImage" src={$patternImagedata1} alt="Patterned" />
                            <button class="button" on:click={() => handleImageMaskSelect(1)}>Select</button>
                        {/if}
                    </div>
                    <div class="image-pane">
                        {#if $patternImagedata2.length === 0 && !loading}
                            <p>Image is not clicked yet!</p>
                        {:else if $patternImagedata2.length !== 0}
                            <img class="patternedImage" src={$patternImagedata2} alt="Patterned" />
                            <button class="button" on:click={() => handleImageMaskSelect(2)}>Select</button>
                        {/if}
                    </div>
                    {#if loading}
                    <div class="loadingContainer">
                        <Shadow size="30" color="var(--color-ultramarine-blue)" unit="px" duration="1s" />
                    </div>
                    {/if}
                </div>
            </div>
		{:else}
			<div class="images-container">
				<div class="image-pane">
					<h5>Click on the image to extract the pattern</h5>
					<img
						class="patternedImage"
						src={$selectedPatternImage}
						alt="Patterned"
						on:click={generatePattern}
					/>
					<div class="slidecontainer">
						<pan>color range</pan>
						<input
							type="range"
							min="1"
							max="3"
							bind:value={color_thre}
							class="slider"
							id="myRange"
						/>
						<pan>{color_thre}</pan>
					</div>
					<button on:click={backToImageMaskSelect} class="button">Back</button>
				</div>
				<div class="image-pane lastImagePane">
					{#if $extractedPatternData.length === 0 && !loading}
						<p>Image is not clicked yet!</p>
					{:else if $extractedPatternData.length !== 0}
						<img class="patternedImage" src={$extractedPatternData} alt="Patterned" />
						<!-- <button class="button" on:click={() => alert("TODO: do the similarity search ...")}>Next</button> -->
					{/if}
					{#if loading}
						<div class="loadingContainer">
							<Shadow size="30" color="var(--color-ultramarine-blue)" unit="px" duration="1s" />
						</div>
					{/if}
				</div>
			</div>
		{/if}
	{:else}
		<div class="upload-container">
			<input type="file" id="file-upload" accept="image/*" on:change={handleFilesChange} />
			<label for="file-upload" class="upload-button"
				>Click to upload or drag and drop an image here</label
			>
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

	.upload-container,
	.images-container {
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

	.image-pane:not(:last-child) {
		border-right: 1px solid #ccc;
	}

	/* .image-pane:last-child{
        border-left: 1px solid #ccc;
		position: relative;
	} */

	.patternedImage {
		padding-bottom: 43px;
		padding-top: 6px;
	}

	.button {
		margin-top: 20px;
	}

	h5 {
		margin-bottom: 10px;
	}

	img {
		max-width: 100%;
		max-height: 400px;
	}

	input[type='file'] {
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

	.loadingContainer {
		background-color: rgba(51, 51, 51, 0.2);
		position: absolute;
		width: 100%;
		height: 100%;
		display: flex;
		justify-content: center;
		align-items: center;
	}

    .outputWrapper{
        flex: 3;
        display: flex;
        flex-shrink: 0;
        position: relative;
    }

    .outputWrapper .image-pane{
        padding-top: 50px;
    }

    .firstImagePane{
        flex: 1;
        flex-shrink: 0;
    }

    .lastImagePane{
        position: relative;
    }
</style>
