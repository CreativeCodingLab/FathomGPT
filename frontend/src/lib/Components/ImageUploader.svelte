<script lang="ts">
	import type { BoundingBox } from '../types/BoundingBoxType';
	import { onMount } from 'svelte';
	import { tick } from 'svelte';

	let fileInput: HTMLInputElement;
	let imgElement: HTMLImageElement;
	let boundingBox: BoundingBox = { x: 50, y: 50, width: 100, height: 100 };
	let lastImageWidth: Number = 0;
	let lastImageHeight: Number = 0;
	let target: HTMLElement | null = null;
	let resizeHandleBottomRight: HTMLElement | null = null;
	let resizeHandleTopLeft: HTMLElement | null = null;
	let resizeHandleTopRight: HTMLElement | null = null;
	let resizeHandleBottomLeft: HTMLElement | null = null;
	let isOpen: Boolean = false;
	let isDragging = false;
	let isResizing = false;
	let resizingCorner = '';
	let isImageBeingCropped = false;
	let startX: number,
		startY: number;

		function onMouseDown(event: MouseEvent | TouchEvent) {
		event.preventDefault();
		const { x, y } = getEventCoordinates(event);
		startX = x;
		startY = y;

		if (event.target === target) {
			isDragging = true;
		} else if (
			event.target === resizeHandleBottomRight ||
			event.target === resizeHandleTopLeft ||
			event.target === resizeHandleTopRight ||
			event.target === resizeHandleBottomLeft
		) {
			isResizing = true;
			resizingCorner = event.target.id;
		}

		document.addEventListener('mousemove', onMouseMove);
		document.addEventListener('mouseup', onMouseUp);
		// Adding touch event listeners
		document.addEventListener('touchmove', onMouseMove, { passive: false });
		document.addEventListener('touchend', onMouseUp);
	}

	function handleDrop(event: MouseEvent) {
    	event.preventDefault();
    	const newFiles = event.dataTransfer.files ? [...event.dataTransfer.files] : [];
		onFileAdded(newFiles[0])
	}

	function handleDragOver(event: MouseEvent) {
		event.preventDefault();
	}

	function handleClick() {
		fileInput.click();
	}

	function onMouseMove(event: MouseEvent | TouchEvent) {
		event.preventDefault();
		const { x, y } = getEventCoordinates(event);
		let dx = x - startX;
		let dy = y - startY;

		if (isDragging) {
			boundingBox.x = Math.min(Math.max(boundingBox.x + dx, 0), imgElement.width - boundingBox.width);
			boundingBox.y = Math.min(Math.max(boundingBox.y + dy, 0), imgElement.height - boundingBox.height);
		} else if (isResizing) {
			switch (resizingCorner) {
				case 'resizeHandleBottomRight':
					boundingBox.width = Math.min(Math.max(boundingBox.width + dx, 10), imgElement.width - boundingBox.x);
					boundingBox.height = Math.min(Math.max(boundingBox.height + dy, 10), imgElement.height - boundingBox.y);
					break;
				case 'resizeHandleTopLeft':
					dx = Math.min(Math.max(dx, -boundingBox.x), boundingBox.width - 10);
					dy = Math.min(Math.max(dy, -boundingBox.y), boundingBox.height - 10);
					boundingBox.x += dx;
					boundingBox.y += dy;
					boundingBox.width -= dx;
					boundingBox.height -= dy;
					break;
				case 'resizeHandleTopRight':
					dy = Math.min(Math.max(dy, -boundingBox.y), boundingBox.height - 10);
					boundingBox.y += dy;
					boundingBox.width = Math.min(Math.max(boundingBox.width + dx, 10), imgElement.width - boundingBox.x);
					boundingBox.height -= dy;
					break;
				case 'resizeHandleBottomLeft':
					dx = Math.min(Math.max(dx, -boundingBox.x), boundingBox.width - 10);
					boundingBox.x += dx;
					boundingBox.width -= dx;
					boundingBox.height = Math.min(Math.max(boundingBox.height + dy, 10), imgElement.height - boundingBox.y);
					break;
			}
		}

		startX = x;
		startY = y;
	}

	function onMouseUp() {
		isDragging = false;
		isResizing = false;
		resizingCorner = '';
		document.removeEventListener('mousemove', onMouseMove);
		document.removeEventListener('mouseup', onMouseUp);
		document.removeEventListener('touchmove', onMouseMove);
    	document.removeEventListener('touchend', onMouseUp);
	}

	function getEventCoordinates(event: MouseEvent | TouchEvent): { x: number; y: number } {
		if (event instanceof MouseEvent) {
			return { x: event.clientX, y: event.clientY };
		} else {
			const touch = event.touches[0] || event.changedTouches[0];
			return { x: touch.clientX, y: touch.clientY };
		}
	}

	onMount(() => {
		document.addEventListener('fileUploader', function (e) {
			//@ts-ignore
			if(e.detail.popupOpened){
				document.body.style="overflow:hidden"
				isOpen = true;
				isImageBeingCropped = false
			}
			//@ts-ignore
			//imgElement.src = e.detail;
		});

		window.addEventListener("resize",()=>{
			if(imgElement!=null){
				boundingBox.x = boundingBox.x*imgElement.width/lastImageWidth
				boundingBox.y = boundingBox.y*imgElement.height/lastImageHeight
				boundingBox.width = boundingBox.width*imgElement.width/lastImageWidth
				boundingBox.height = boundingBox.height*imgElement.height/lastImageHeight

				lastImageWidth = imgElement.width
				lastImageHeight = imgElement.height
			}
		})
	});

	async function sendCroppedImage() {
		if (!imgElement) return;

		const scaleX = imgElement.naturalWidth / imgElement.width;
		const scaleY = imgElement.naturalHeight / imgElement.height;

		const offscreenCanvas = document.createElement('canvas');
		offscreenCanvas.width = boundingBox.width;
		offscreenCanvas.height = boundingBox.height;
		const offCtx = offscreenCanvas.getContext('2d');

		if (!offCtx) return;

		// Draw the cropped area on the in-memory canvas
		// Use natural dimensions for source coordinates and dimensions
		offCtx.drawImage(
			imgElement,
			boundingBox.x * scaleX, // Scale the x coordinate
			boundingBox.y * scaleY, // Scale the y coordinate
			boundingBox.width * scaleX, // Scale the width
			boundingBox.height * scaleY, // Scale the height
			0,
			0,
			boundingBox.width,
			boundingBox.height
		);

		offscreenCanvas.toBlob(async (blob) => {
			const blobUrl = URL.createObjectURL(blob);
			console.log(blobUrl);
			const myCustomEvent = new CustomEvent('imageSelected', {
				detail: blobUrl
			});
			document.dispatchEvent(myCustomEvent);
			closeModal();
		});
	}

	function closeModal() {
		document.body.style="overflow:auto"
		isOpen = false;
	}

	function bgClicked(event: MouseEvent): void {
		if (event.target === backgroundRef) {
			closeModal();
		}
	}

	function handleFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) {
			console.error('No file selected.');
			return;
		}
		onFileAdded(input.files[0])

	}

	async function selectImage(event: Event,x: Number,y: Number,width: Number,height: Number){
		isImageBeingCropped = true
		await tick();
		imgElement.src = event.target.getAttribute("src")
		boundingBox.x=x*imgElement.width
		boundingBox.y=y*imgElement.height
		boundingBox.width=width*imgElement.width
		boundingBox.height=height*imgElement.height
		lastImageWidth = imgElement.width
		lastImageHeight = imgElement.height
	}

	async function onFileAdded(file: any) {
		const reader = new FileReader();
		reader.onload = async (e: ProgressEvent<FileReader>) => {
			if(reader && reader.result){
				isImageBeingCropped = true
				await tick();
				imgElement.src = reader.result;
				BoundingBox = { x: 50, y: 50, width: 100, height: 100 }
				lastImageWidth = imgElement.width
				lastImageHeight = imgElement.height
			}
			fileInput.value=null
		};
		reader.onerror = (error) => alert("Error reading the file! " + error);
		reader.readAsDataURL(file);
	}

	let backgroundRef: Element;
</script>

<main>
	<div class="imageDetailsOuterContainer" class:active={isOpen}>
		<div class="imageDetailsContainerWrapper" on:click={bgClicked} bind:this={backgroundRef}>
			<div class="imageDetailsContainer">
				<div class="header">
					<h1>Upload Image</h1>
					<button class="closeBtn" on:click={closeModal}><i class="fa-solid fa-xmark" /></button>
				</div>
				{#if !isImageBeingCropped}
				<div
					class="drop-zone"
					on:click={handleClick}
					on:dragover|preventDefault={handleDragOver}
					on:dragenter={e => e.currentTarget.classList.add('drag-over')}
					on:dragleave={e => e.currentTarget.classList.remove('drag-over')}
					on:drop={handleDrop}>
					Click or drag and drop an image here to upload
					</div>
				<input type="file" id="file-upload" hidden accept="image/*" bind:this={fileInput} on:change={handleFileChange} />
				<h3>Sample Images to Try</h3>
				<div class="sampleImagesContainer">
					<div class="sampleImage">
						<img src="./sample-image-2.jpg" on:click={(e)=>selectImage(e,  0.06310446174545778, 0.1284144744025619, 0.8748704184266516, 0.825471039873701)}/>
					</div>
					<div class="sampleImage">
						<img src="./sample-image-3.jpeg" on:click={(e)=>selectImage(e, 0.2522586299892125, 0.2659940730410529, 0.479165645124383, 0.5166005694529184)}/>
					</div>
					<div class="sampleImage">
						<img src="./sample-image-1.jpg" on:click={(e)=>selectImage(e,  0.44362226635284885, 0.3705395275865075, 0.48359746330620124, 0.3752672361195851)}/>
					</div>
				</div>
				{:else}
					<div class="infoText">
						Please move/resize the red box to select the portion of the image that contains the specimen.
					</div>
					<div class="detailsContainer">
						<img alt="Loaded image" bind:this={imgElement} />
						<div
							id="target"
							bind:this={target}
							style="width: {boundingBox.width}px; height: {boundingBox.height}px; background: rgba(255, 0, 0, 0.4); position: absolute; left: {boundingBox.x}px; top: {boundingBox.y}px"
							on:mousedown={onMouseDown}
							on:touchstart={onMouseDown}
						>
							<button
								class="resizeHandle handleBottomRight"
								id="resizeHandleBottomRight"
								bind:this={resizeHandleBottomRight}
								on:mousedown={onMouseDown}
								on:touchstart={onMouseDown}
							/>
							<button
								class="resizeHandle handleTopLeft"
								id="resizeHandleTopLeft"
								bind:this={resizeHandleTopLeft}
								on:mousedown={onMouseDown}
								on:touchstart={onMouseDown}
							/>
							<button
								class="resizeHandle handleTopRight"
								id="resizeHandleTopRight"
								bind:this={resizeHandleTopRight}
								on:mousedown={onMouseDown}
								on:touchstart={onMouseDown}
							/>
							<button
								class="resizeHandle handleBottomLeft"
								id="resizeHandleBottomLeft"
								bind:this={resizeHandleBottomLeft}
								on:mousedown={onMouseDown}
								on:touchstart={onMouseDown}
							/>
						</div>
					</div>
					<button class="button imgUploadBtn" on:click={sendCroppedImage}>Add</button>
				{/if}
			</div>
		</div>
	</div>
</main>

<style>
	.closeBtn:hover {
		background: rgba(0, 0, 0, 0.2);
	}
	.closeBtn:active {
		background: rgba(0, 0, 0, 0.3);
	}

	.imageDetailsContainer .header {
		display: flex;
		justify-content: space-between;
	}

	.imageDetailsOuterContainer {
		position: fixed;
		left: 0;
		top: 0;
		width: 100%;
		min-height: 100vh;
		z-index: 1000;
		background-color: rgba(0, 0, 0, 0.5);
		opacity: 0;
		pointer-events: none;
		transition: 0.2s;
		backdrop-filter: blur(1px);
		overflow: auto;
		height: 100vh;
	}
	.imageDetailsOuterContainer.active {
		opacity: 1;
		pointer-events: all;
	}

	.imageDetailsContainerWrapper {
		padding: 20px;
		display: flex;
		justify-content: center;
		align-items: center;
		min-height: 100vh;
	}

	.imageDetailsContainer {
		padding: 20px;
		max-width: 1200px;
		border-radius: 10px;
		background: white;
		display: flex;
		flex-direction: column;
	}

	.detailsContainer {
		display: flex;
		flex-direction: row;
		margin: 10px 0px 30px 0px;
		flex-direction: column;
		position: relative;
	}

	.detailsContainer img {
		user-select: none;
	}

	.closeBtn {
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
	.closeBtn:hover {
		background: rgba(0, 0, 0, 0.2);
	}
	.closeBtn:active {
		background: rgba(0, 0, 0, 0.3);
	}

	.imgUploadBtn {
		width: fit-content;
		margin: 0px auto;
	}

	.infoText {
		padding: 5px 0px 0px 0px;
		color: var(--color-pelagic-gray);
	}

	.resizeHandle {
		width: 14px;
		height: 14px;
		background: var(--accent-color);
		position: absolute;
		border-radius: 50%;
		border: 2px solid white;
		box-sizing: border-box;
		transition: 0.2s ease;
	}
	.resizeHandle:hover {
		background-color: white;
		border: 4px solid var(--accent-color);
	}
	.resizeHandle:active {
		transform: scale(1.5);
		transform-origin: center center;
	}
	img {
		max-height: calc(100vh - 240px);
		max-width: calc(100vw - 70px);
	}

	.handleBottomRight {
		right: -6px;
		bottom: -6px;
		cursor: se-resize;
	}

	.handleTopLeft {
		left: -6px;
		top: -6px;
		cursor: nw-resize;
	}

	.handleTopRight {
		right: -6px;
		top: -6px;
		cursor: ne-resize;
	}

	.handleBottomLeft {
		left: -6px;
		bottom: -6px;
		cursor: sw-resize;
	}

	#target {
		cursor: move;
		border: 2px solid rgba(255, 0, 0, 0.7);
		box-sizing: border-box;
	}

	.upload-container{
		width: 95%;
		max-width: 1200px;
		display: flex;
		justify-content: space-around;
	}

	.drop-zone {
		border: 2px dashed #ccc;
		padding: 20px;
		text-align: center;
		margin: 20px 0px;
		cursor: pointer;
		min-height: 120px;
		display: flex;
		justify-content: center;
		align-items: center;
	}
	.drop-zone:hover{
		background-color: rgba(0, 0, 0, 0.05);
	}
	.drop-zone:active{
		background-color: rgba(0, 0, 0, 0.08);
	}

	.sampleImagesContainer {
		display: grid;
		gap: 10px;
		justify-items: center;
		align-items: start;
		margin-top: 10px;
		grid-template-columns: 1fr 1fr 1fr 1fr;
		@media (max-width: 864px) {
			grid-template-columns: 1fr 1fr;
		}
		@media (max-width: 600px) {
			grid-template-columns: 1fr;
		}
	}

	.sampleImage {
		border-radius: 5px;
		overflow: hidden;
		cursor: pointer;
		display: flex;
		justify-content: center;
		align-items: center;
		width: 100%;
		height: 100%;
		min-width: 250px;
		height: 180px;
		justify-content: center;
		align-items: center;
		background-color: black;
	}

	.sampleImage img{
		transition: 0.2s ease;
		max-width: 100%;
		max-height: 100%;
		width: auto;
		height: auto;
	}

	.sampleImage:hover img {
		transform: scale(1.1);
		transform-origin: center center;

	}

</style>
