# 🔢 Đồ án cá nhân: 8-Puzzle Solver

## 🎯 Mục tiêu
Xây dựng một chương trình giải bài toán **8-Puzzle** sử dụng nhiều thuật toán tìm kiếm khác nhau trong lĩnh vực Trí tuệ nhân tạo.

---

## 🧠 Các thuật toán được triển khai

| Thuật Toán               | Mô Tả                                                                 | Minh Hóa GIF                              |
|--------------------------|----------------------------------------------------------------------|-------------------------------------------|
| **Breadth-First Search (BFS)** | Khám phá tuần tự theo từng lớp: Bắt đầu từ nút gốc, thuật toán duyệt qua tất cả các nút ở cùng độ sâu trước khi chuyển sang độ sâu tiếp theo. Ưu điểm: Đảm bảo tìm được đường đi ngắn nhất (nếu có). Nhược điểm: Tốn nhiều bộ nhớ cho các bài toán có không gian trạng thái rộng.             | <img src="images/bfs.gif" width="500" alt="BFS"> |
| **Depth-First Search (DFS)**   | Đi sâu vào một nhánh trước khi quay lui: Thuật toán khám phá một nhánh của cây tìm kiếm càng sâu càng tốt trước khi quay lại và khám phá các nhánh khác. Ưu điểm: Tiết kiệm bộ nhớ hơn BFS. Nhược điểm: Không đảm bảo tìm được đường đi ngắn nhất và có thể bị mắc kẹt trong các nhánh vô hạn.    | <img src="images/dfs.gif" width="500" alt="DFS"> |
| **Uniform Cost Search (UCS)**  | Mở rộng nút có chi phí đường đi thấp nhất: Tương tự BFS, nhưng ưu tiên các đường đi có tổng chi phí từ nút gốc đến nút hiện tại là nhỏ nhất. Ưu điểm: Tìm được đường đi có chi phí thấp nhất. Nhược điểm: Có thể tốn kém thời gian nếu hàm chi phí không tốt.        | <img src="images/ucs.gif" width="500" alt="UCS"> |
| **Iterative Deepening DFS (IDDFS)** | Kết hợp sức mạnh của DFS và BFS: Thực hiện DFS với độ sâu giới hạn tăng dần. Ưu điểm: Vừa tìm được đường đi ngắn nhất (như BFS) vừa tiết kiệm bộ nhớ (như DFS). Nhược điểm: Có thể tính toán lại các trạng thái ở các độ sâu khác nhau.                 | <img src="images/iddfs.gif" width="500" alt="IDDFS"> |
| **Greedy Best-First Search**   | Ưu tiên khám phá các nút có vẻ "hứa hẹn" nhất: Sử dụng hàm heuristic để ước tính chi phí từ nút hiện tại đến mục tiêu và chọn nút có giá trị heuristic thấp nhất để mở rộng. Ưu điểm: Thường tìm được giải pháp nhanh chóng. Nhược điểm: Có thể không tìm được đường đi tối ưu nếu hàm heuristic không chính xác.             | <img src="images/greedy.gif" width="500" alt="GREEDY"> |
| **A* Search**                 | Cân bằng giữa chi phí đã đi và ước tính chi phí còn lại: Kết hợp chi phí thực tế từ nút gốc đến nút hiện tại (g(n)) và chi phí ước tính từ nút hiện tại đến mục tiêu (h(n)) để đánh giá (f(n)=g(n)+h(n)). Ưu điểm: Tìm được đường đi ngắn nhất một cách hiệu quả nếu hàm heuristic chấp nhận được. Nhược điểm: Tốn bộ nhớ để lưu trữ các nút đã xét.        | <img src="images/astar.gif" width="500" alt="A*"> |
| **IDA* Search**               | Phiên bản tiết kiệm bộ nhớ của A:* Thực hiện tìm kiếm theo chiều sâu lặp đi lặp lại với ngưỡng chi phí (f(n)) tăng dần. Ưu điểm: Tiết kiệm bộ nhớ hơn A* mà vẫn đảm bảo tìm được đường đi tối ưu. Nhược điểm: Có thể tốn thời gian hơn A* do phải thực hiện nhiều lần tìm kiếm DFS.                     | <img src="images/idastar.gif" width="500" alt="IDA*"> |
| **Simple Hill Climbing**       | Chỉ di chuyển đến trạng thái lân cận tốt hơn: Bắt đầu từ một trạng thái ngẫu nhiên và liên tục di chuyển đến trạng thái lân cận có giá trị hàm mục tiêu tốt hơn. Ưu điểm: Đơn giản và dễ thực hiện. Nhược điểm: Dễ bị mắc kẹt ở cực trị cục bộ (local optima).                       | <img src="images/simplehillclimbing.gif" width="500" alt="Simple HC"> |
| **Steepest Hill Climbing**     | Xem xét tất cả các trạng thái lân cận và chọn trạng thái tốt nhất: Tại mỗi bước, thuật toán đánh giá tất cả các trạng thái lân cận và di chuyển đến trạng thái có giá trị hàm mục tiêu tốt nhất. Ưu điểm: Cải thiện hơn Simple Hill Climbing trong việc tránh các bước đi tồi tệ. Nhược điểm: Vẫn có khả năng bị mắc kẹt ở cực trị cục bộ.     | <img src="images/steepesthillclimbing.gif" width="500" alt="Steepest HC"> |
| **Stochastic Hill Climbing**   | Cho phép di chuyển đến trạng thái xấu với một xác suất nhất định: Tương tự Steepest Hill Climbing, nhưng có thêm cơ chế ngẫu nhiên để thoát khỏi cực trị cục bộ bằng cách đôi khi chấp nhận các bước đi "xuống dốc". Ưu điểm: Ít bị mắc kẹt ở cực trị cục bộ hơn các thuật toán leo dốc đơn giản. Nhược điểm: Vẫn không đảm bảo tìm được giải pháp tối ưu toàn cục.                | <img src="images/stochastichillclimbing.gif" width="500" alt="Stochastic HC"> |
| **Simulated Annealing**        | Mô phỏng quá trình làm nguội kim loại: Bắt đầu với "nhiệt độ" cao và giảm dần theo thời gian. Ở nhiệt độ cao, thuật toán có khả năng chấp nhận các giải pháp xấu hơn để thoát khỏi cực trị cục bộ. Khi nhiệt độ giảm, khả năng này giảm dần. Ưu điểm: Có khả năng tìm được giải pháp tối ưu toàn cục tốt hơn các thuật toán leo dốc khác. Nhược điểm: Yêu cầu điều chỉnh các tham số (lịch làm nguội) cẩn thận để đạt hiệu quả tốt.    | <img src="images/simulatedannealing.gif" width="500" alt="Simulated Annealing"> |
| **Beam Search**                | Duy trì một "chùm" các trạng thái tốt nhất: Thay vì chỉ giữ lại một trạng thái duy nhất, thuật toán giữ lại một số lượng cố định (kích thước chùm) các trạng thái có vẻ hứa hẹn nhất ở mỗi bước. Ưu điểm: Khám phá không gian trạng thái rộng hơn so với các thuật toán tìm kiếm cục bộ đơn lẻ. Nhược điểm: Có thể bỏ lỡ giải pháp tối ưu nếu kích thước chùm quá nhỏ.   | <img src="images/beamsearch.gif" width="500" alt="Beam Search"> |

## 👨‍💻 Tác giả

**Trần Thành Trung**  
MSSV: `23110351`  
Môn: `Trí Tuệ Nhân Tạo`
Giáo viên hướng dẫn: `Phan Thị Huyền Trang` 

---
