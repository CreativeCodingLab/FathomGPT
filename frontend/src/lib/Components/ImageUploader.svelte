<script lang="ts">
	import type { BoundingBox } from '../types/BoundingBoxType';
	import { onMount } from 'svelte';

	let imgElement: HTMLImageElement;
	let boundingBox: BoundingBox = { x: 50, y: 50, width: 100, height: 100 };
	let target: HTMLElement | null = null;
    let resizeHandle: HTMLElement | null = null;
	let isOpen: Boolean = false;
	let isDragging = false;
	let isResizing = false;
	let startX: number, startY: number, startWidth:number, startHeight: number;

	function onMouseDown(event: MouseEvent) {
		if (event.target === target) {
			isDragging = true;
		} else if (event.target === resizeHandle) {
			isResizing = true;
		}
		startX = event.clientX;
		startY = event.clientY;
		startWidth = boundingBox.width;
		startHeight = boundingBox.height;
		document.addEventListener('mousemove', onMouseMove);
		document.addEventListener('mouseup', onMouseUp);
	}

	function onMouseMove(event: MouseEvent) {
		if (isDragging) {
			const dx = event.clientX - startX;
			const dy = event.clientY - startY;
			boundingBox.x += dx;
			boundingBox.y += dy;
			startX = event.clientX;
			startY = event.clientY;
		} else if (isResizing) {
			const dx = event.clientX - startX;
			const dy = event.clientY - startY;
			boundingBox.width = Math.max(50, startWidth + dx); // Minimum size = 50
			boundingBox.height = Math.max(50, startHeight + dy); // Minimum size = 50
		}
	}

	function onMouseUp() {
		isDragging = false;
		isResizing = false;
		document.removeEventListener('mousemove', onMouseMove);
		document.removeEventListener('mouseup', onMouseUp);
	}

    onMount(() => {
        document.addEventListener('imageAdded', function (e) {
            //@ts-ignore
            imgElement.src=e.detail
            isOpen = true;
        });
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
        console.log(blobUrl)
        const myCustomEvent = new CustomEvent('imageSelected', {
            detail: blobUrl
        });
        document.dispatchEvent(myCustomEvent);
        closeModal();
    });
}


	function closeModal() {
        isOpen = false;
    }
</script>

<main>
	<div class="imageDetailsOuterContainer" class:active={isOpen}>
		<div class="imageDetailsContainer">
			<div class="header">
				<h1>Upload Image</h1>
				<button class="closeBtn" on:click={closeModal}><i class="fa-solid fa-xmark" /></button>
			</div>
			<div class="infoText">
				Please move/resize the box to select the portion of the image that contains the specimen.
			</div>
            <div class="detailsContainer">
                <img alt="Loaded image" bind:this={imgElement}/>
                <div id="target" bind:this={target} 
                     style="width: {boundingBox.width}px; height: {boundingBox.height}px; background: rgba(255, 0, 0, 0.5); position: absolute; left: {boundingBox.x}px; top: {boundingBox.y}px" 
                     on:mousedown={onMouseDown}>
                    <div id="resizeHandle" class="resizeHandle" bind:this={resizeHandle} on:mousedown={onMouseDown}></div> <!-- Bind resizeHandle here -->
                </div>
            </div>
			<button class="button imgUploadBtn" on:click={sendCroppedImage}>Add</button>
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
	.imageDetailsOuterContainer.active {
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

	.detailsContainer {
		display: flex;
		flex-direction: row;
		padding: 10px 0px 30px 0px;
		flex-direction: column;
		position: relative;
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
        width: 10px;
        height: 10px;
        background: var(--accent-color);
        position: absolute;
        right: -3px;
        bottom: -3px;
        cursor: se-resize;
        border-radius: 50%;
    }
	img{
		max-height: calc(100vh - 220px);
	}
</style>
