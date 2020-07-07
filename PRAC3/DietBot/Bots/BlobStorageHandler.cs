using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;

using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace Microsoft.BotBuilderSamples.Bots
{
    public class BlobStorageHandler
    {
        private string connectionString;
        private string containerName;

        private BlobServiceClient blobServiceClient;
        private BlobContainerClient containerClient;
        public void Init()
        {
            connectionString = "";
            blobServiceClient = new BlobServiceClient(connectionString);
            containerName = "food101";

            containerClient = blobServiceClient.GetBlobContainerClient(containerName);
        }

        public async Task Upload(string from, string to)
        {
            BlobClient blobClient = containerClient.GetBlobClient(to);

            using FileStream uploadFileStream = File.OpenRead(from);
            await blobClient.UploadAsync(uploadFileStream, true);
            uploadFileStream.Close();
        }
        public async Task Download(string from, string to)
        {
            BlobClient blobClient = containerClient.GetBlobClient(from);
            BlobDownloadInfo download = await blobClient.DownloadAsync();

            using (FileStream downloadFileStream = File.OpenWrite(to))
            {
                await download.Content.CopyToAsync(downloadFileStream);
                downloadFileStream.Close();
            }
        }

        public async Task<bool> Exists(string blob)
        {
            BlobClient blobClient = containerClient.GetBlobClient(blob);
            var res = await blobClient.ExistsAsync();

            return res.Value;
        }

        public async Task<List<string>> List(string prefix = "")
        {
            List<string> list = new List<string>();
            var blobs = containerClient.GetBlobsAsync(prefix: prefix);
            await foreach (BlobItem blobItem in blobs)
            {
                list.Add(blobItem.Name);
            }

            return list;
        }
    }
}