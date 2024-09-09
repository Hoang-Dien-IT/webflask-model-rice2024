function validateForm() {
    const inputs = document.querySelectorAll('input[type="text"]');
    for (let i = 0; i < inputs.length; i++) {
        const value = inputs[i].value;
        if (!/^\d+(\.\d+)?$/.test(value)) {
            alert('Vui lòng nhập đúng ' + inputs[i].name);
            return false; // Ngăn gửi form nếu có lỗi
        }
    }
    return true; // Cho phép gửi form nếu tất cả hợp lệ
}
