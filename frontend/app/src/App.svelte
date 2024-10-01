<script>
  let images = [];

  async function fetchImages() {
    const response = await fetch("http://localhost:5001/images");
    images = await response.json();
  }

  fetchImages();
</script>

<div class="image-grid">
  {#each images as image}
    <img src={`http://localhost:5001/photos/${image}`} alt={image} />
  {/each}
</div>

<style>
  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
  }
  img {
    width: 100%;
    object-fit: cover;
  }
</style>
