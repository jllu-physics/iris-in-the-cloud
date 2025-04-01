### 🌸 iris-in-the-cloud ☁️

**End-to-end deep learning model deployment on a budget**

This project demonstrates how to deploy a deep learning model with minimal cost and complexity:

* 🚀 FastAPI handles HTTP requests with ease and speed

* 📦 Docker enables simple, reproducible containerization

* 🧠 TensorFlow powers a compact, fully-functional DNN trained on the classic Iris dataset

* ☁️ AWS EC2 hosts the model via a lightweight, always-on instance

* 💸 Total cost: ~ $10/month

The inference endpoint is live [here](http://3.17.238.30:8000/) — have fun! 🎯

#### Reproducing Results

To replicate the training process, including hyperparameter selection, run

```bash
bash run.sh
```

#### Deploy Steps

1. **Launch AWS EC2 Instance**
   - Instance type: `t3.small` (should suffice for lightweight inference)
   - Ensure security group allows inbound access on port 8000 (or 80/443 if you're proxying)

2. **Copy project assets to EC2**
   ```bash
   scp -i your-key.pem -r ./your-project ec2-user@<EC2-IP>:/home/ec2-user/
   ```

3. **Install Docker**
   ```bash
   sudo yum update -y
   sudo amazon-linux-extras enable docker
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user  # optional: run docker without sudo after logout/login
   ```

4. **Build Docker Image**
   ```bash
   cd your-project
   sudo docker build -t infer_image .
   ```

5. **Run Docker Container**
   ```bash
   sudo docker run -d \
       -p 8000:8000 \
       --name infer_container \
       --restart=always \
       --memory="1.5g" \
       --cpus="1.5" \
       infer_image
   ```

   - `--restart=always`: ensures container is restarted if it crashes or the EC2 instance reboots.
   - `--memory` and `--cpus`: prevent your container from overloading the EC2 instance under traffic spikes.

6. **Verify Inference Endpoint**

    After deploying, verify the app is running correctly:

    - Open your browser and visit:
    `http://<your-ec2-ip>:8000/status`

    - Or use `curl` from your local terminal:
    ```bash
    curl http://<your-ec2-ip>:8000/status
