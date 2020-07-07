using System.Threading.Tasks;

using System.Net.Http;
using System.Net.Http.Headers;
using System.IO;
using Newtonsoft.Json;

namespace Microsoft.BotBuilderSamples.Bots
{
    public struct RecognitionResult
    {
        public string tag { get; set; }
        public float prob { get; set; }

        public RecognitionResult(string t, float p)
        {
            tag = t;
            prob = p;
        }
    }
    public class FoodRecognizer
    {
        public HttpClient client;
        public string url;

        public FoodRecognizer()
        {
            client = new HttpClient();
            client.DefaultRequestHeaders.Add("Prediction-Key", "");
            url = "";
        }

        public async Task<RecognitionResult> MakePredictionRequest(string imageFilePath)
        {
            HttpResponseMessage response;

            byte[] byteData = GetImageAsByteArray(imageFilePath);

            using (var content = new ByteArrayContent(byteData))
            {
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                response = await client.PostAsync(url, content);

                string strRes = await response.Content.ReadAsStringAsync();
                dynamic res = (dynamic)JsonConvert.DeserializeObject(strRes);

                string tag = res.predictions[0].tagName;
                float prob = res.predictions[0].probability;

                return new RecognitionResult(tag, prob);
            }
        }

        private static byte[] GetImageAsByteArray(string imageFilePath)
        {
            FileStream fileStream = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read);
            BinaryReader binaryReader = new BinaryReader(fileStream);
            return binaryReader.ReadBytes((int)fileStream.Length);
        }
    }
}