using System.Collections.Generic;
using System.IO;
using CsvHelper;
using System.Linq;
using System.Globalization;

namespace Microsoft.BotBuilderSamples.Bots
{
    public class Record
    {
        public string product_name {get; set;}
        public float calories {get; set;}
        public float fat {get; set;}
        public float cholesterol {get; set;}
        public float sodium {get; set;}
        public float carbohydrate {get; set;}
        public float protein {get; set;}
    }

    public class PersonalDataRecord
    {
        public float height {get; set;}
        public float weight {get; set;}
        public int age {get; set;}
        public string sex {get; set;}
        public float phys_act_coef {get; set;}
        public float caloriesNorm {get; set;}
        public float proteinNorm {get; set;}
        public float carbohydrateNorm {get; set;}
        public float fatNorm {get; set;}

        public void ComputeNorms(){
            float BMR = 9.99f * weight + 6.25f * height - 4.92f * age;
            if(sex == "male"){
                BMR -= 161;
            }
            else{
                BMR += 5;
            }
            caloriesNorm = phys_act_coef * BMR;
            proteinNorm = 1.75f * weight;
            fatNorm = 1.15f * weight;
            carbohydrateNorm = 2.0f * weight;
        }
    }

    public class EatingRecord{
        public string date {get; set;}
        public string eating {get; set;}
        public string product_name {get; set;}
        public float calories {get; set;}
        public float fat {get; set;}
        public float cholesterol {get; set;}
        public float sodium {get; set;}
        public float carbohydrate {get; set;}
        public float protein {get; set;}
    }

    static class CSVHandler{
        public static void WriteToCsv<T>(List<T> list, string path)
        {
            using (var writer = new StreamWriter(path))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(list);
            }
        }

        public static List<T> ReadFromCsv<T>(string path)
        {
            List<T> res;
            using (var reader = new StreamReader(path))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                res = csv.GetRecords<T>().ToList();
            }
            return res;
        }

        public static void AppendToCsv<T>(List<T> list, string path)
        {
            var r = ReadFromCsv<T>(path);
            r.AddRange(list);
            WriteToCsv(r, path);
        }
    }
}
    