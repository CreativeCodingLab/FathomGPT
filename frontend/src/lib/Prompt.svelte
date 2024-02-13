<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {Shadow} from 'svelte-loading-spinners';
	import { onMount } from 'svelte';
	import { tick } from 'svelte';

	const dispatch = createEventDispatcher();

	let value = '';
	let placeholder = 'Type your prompt here';
	let heightModifier = 3.2;
	let textarea: HTMLTextAreaElement;
	let loading = false;
	let imageTag: HTMLInputElement;
	let isImageSelected = false;
	let selectedImage: HTMLImageElement;

	export function toggleLoading() {
		loading = !loading
		console.log('toggling loading to ', loading);
	}
	//@ts-ignore
	function convertBlobUrlToBase64(blobUrl, callback) {
		fetch(blobUrl)
			.then(response => {
				if (response.ok) return response.blob();
				throw new Error('Network response was not ok.');
			})
			.then(blob => {
				const reader = new FileReader();
				reader.readAsDataURL(blob);
				reader.onloadend = function() {
					callback(reader.result);
				};
			})
			.catch(error => {
				console.error('There was a problem with the fetch operation:', error);
			});
	}

	function submitPrompt() {
		if (value !== '') {
			if(isImageSelected){
				convertBlobUrlToBase64(selectedImage.src, (imageData: any)=>{
					dispatch('submit', {
						value,
						image: imageData
					});
				})
			}
			else{
				dispatch('submit', {
					value,
				});
			}
		}
	}

	onMount(() => {
        document.addEventListener('imageSelected', async function (e) {
            //@ts-ignore
			isImageSelected = true
			await tick();
            selectedImage.src=e.detail
        });
    });

	function detectEnterPress(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitPrompt();
			textarea.value = '';
			heightModifier = 3.2;
			textarea.focus();
			isImageSelected=false;
		} else if (e.key === 'Enter' && e.shiftKey) {
			heightModifier += 1;
		}
	}

	async function handleFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) {
			console.error('No file selected.');
			return;
		}

		const file = input.files[0];
		const reader = new FileReader();

		reader.onload = (e: ProgressEvent<FileReader>) => {
			if(reader && reader.result){
				const myCustomEvent = new CustomEvent('imageAdded', {
					detail: reader.result as string
				});
				document.dispatchEvent(myCustomEvent);
			}
		};
		reader.onerror = (error) => alert("Error reading the file! " + error);
		reader.readAsDataURL(file);
	}

	function openFileSelector(){
		imageTag.click()
	}

	function removeSelectedImage(){
		isImageSelected=false;
	}

</script>

<main>
	{#if loading}
	<Shadow size="30" color="var(--color-ultramarine-blue)" unit="px" duration="1s" />
	{/if}
	<div class="divWrapper">
		<!-- svelte-ignore a11y-autofocus -->
		{#if isImageSelected}
		<div class="selectedImageWrapper">
			<img class="selectedImage" bind:this={selectedImage}/>
			<button class="removeImageBtn" on:click={removeSelectedImage}><i class="fa-solid fa-xmark" /></button>
		</div>
		{:else}
		<button class="buttonCircled" on:click={openFileSelector}>
			<i class="fa-solid fa-file-import"></i>
		</button>
		{/if}
		<input hidden accept="image/*" type="file" bind:this={imageTag} on:input={handleFileChange}/>
		<textarea
			bind:this={textarea}
			bind:value
			autofocus
			{placeholder}
			on:keypress={detectEnterPress}
			style="--heightModifier: {heightModifier}"
		/>
		<button class="sendBtn" on:click={submitPrompt}>
			<img src="/submit.svg" alt="submission arrow" />
		</button>
	</div>
</main>

<style>
	main {
		position: sticky;
		width: 100%;
		margin: 0 auto;
		bottom: 0;
		left: 0;
		display: grid;
		place-items: center;
		min-height: 10rem;
		overscroll-behavior: none;
		z-index: 20;
		background-image: linear-gradient(
			180deg,
			rgba(0, 0, 0, 0) 0%,
			var(--color-white) 50%,
			var(--color-white) 100%
		);
		padding: 0px var(--chat-padding);
	}
	.divWrapper {
		place-self: center;
		width: 70%;
		border: none;
		border-radius: 0.5rem;
		padding: 0.5rem 0.8rem;
		background-color: var(--color-white);
		color: var(--accent-dark);
		display: flex;
		align-items: center;
		justify-content: center;
		box-shadow: 0 0 2rem rgba(255, 255, 255, 0.2);
		border: 2px solid var(--color-salt-marsh-gray);
		border-radius: 33px;
	}
	.divWrapper:hover{
		border-color: var(--accent-color);
		background-color: var(--color-white);
	}
	.divWrapper:focus-within{
		border-color: var(--accent-color);
		background-color: var(--color-white);
	}
	textarea {
		width: 100%;
		height: calc(var(--heightModifier) * 1rem);
		padding-top: 1rem;
		padding-left: 1rem;
		font-size: 1rem;
		background-color: transparent;
		border: none;
		resize: none;
		display: grid;
		place-items: center;
	}
	textarea:focus {
		outline: none;
	}
	.sendBtn {
		background-color: rgba(255, 255, 255, 0.5);
		padding: 0.2rem;
		border-radius: 0.2rem;
		width: 2.5rem;
		height: 2.5rem;
		border: none;
		place-self: center end;
	}
	.sendBtn:hover {
		background-color: rgba(255, 255, 255, 0.8);
	}
	.sendBtn:active {
		background-color: rgba(0, 0, 0, 0.5);
	}

	img {
		width: 100%;
		height: 100%;
	}
	.selectedImage{
		max-width: 80px;
		max-height: 80px;
		width: auto;
		height: auto;
		border-radius: 10px;
		margin-left: 10px;
	}

	.removeImageBtn {
		width: 24px;
		height: 24px;
		border-radius: 50%;
		border: 0;
		margin-bottom: 10px;
		background: white;
		cursor: pointer;
		align-self: flex-end;
		display: flex;
		justify-content: center;
		align-items: center;
		position: absolute;
		right: -6px;
		top: -6px;
	}
	.removeImageBtn:hover {
		background: #d1d1d1;
	}
	.removeImageBtn:active {
		background: #b1b1b1;
	}

	.selectedImageWrapper{
		position: relative;
	}

</style>
