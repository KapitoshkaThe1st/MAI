using System.Threading.Tasks;

using Microsoft.Azure.CognitiveServices.Knowledge.QnAMaker;
using Microsoft.Azure.CognitiveServices.Knowledge.QnAMaker.Models;

namespace Microsoft.BotBuilderSamples.Bots
{
    public class QnAHandler
    {
        public string authoringKey;
        public string resourceName;

        public string authoringURL;
        public string queryingURL;
        
        public string primaryQueryEndpointKey;

        public string kbId = "";

        QnAMakerRuntimeClient runtimeClient;

        public QnAHandler()
        {
            authoringKey = "";
            resourceName = "qnadietbot1";

            authoringURL = $"https://{resourceName}.cognitiveservices.azure.com";
            queryingURL = $"https://{resourceName}.azurewebsites.net";

            primaryQueryEndpointKey = "";

            runtimeClient = new QnAMakerRuntimeClient(new EndpointKeyServiceClientCredentials(primaryQueryEndpointKey))
            { RuntimeEndpoint = queryingURL };
        }

        public async Task<string> GenerateAnswer(string question)
        {
            var response = await runtimeClient.Runtime.GenerateAnswerAsync(kbId, new QueryDTO { Question = question });
            return response.Answers[0].Answer;
        }
    }
}